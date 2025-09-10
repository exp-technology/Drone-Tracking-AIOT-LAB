"""
ByteTrack
"""

import numpy as np
from collections import deque
from .basetrack import BaseTrack, TrackState
from .tracklet import Tracklet, Tracklet_w_reid
from .matching import *

# for reid
import torch
import torchvision.transforms as T
from .reid_models.engine import load_reid_model, crop_and_resize

# base class
from .basetracker import BaseTracker

class ByteTracker(BaseTracker):
    def __init__(self, args, frame_rate=30):
        
        super().__init__(args, frame_rate=frame_rate)

        # whether to use reid 
        self.with_reid = args.reid
        self.reid_model = None
        if self.with_reid:
            self.reid_model = load_reid_model(args.reid_model, args.reid_model_path, 
                                              device=args.device, trt=args.trt, crop_size=args.reid_crop_size)

        # once init, clear all trackid count to avoid large id
        BaseTrack.clear_count()


    def update(self, output_results, img, ori_img):
        """
        output_results: processed detections (scale to original size) tlbr format
        """

        self.frame_id += 1
        activated_tracklets = []
        refind_tracklets = []
        lost_tracklets = []
        removed_tracklets = []

        scores = output_results[:, 4]
        bboxes = output_results[:, :4]
        categories = output_results[:, -1]

        remain_inds = scores > self.args.conf_thresh
        inds_low = scores > self.args.conf_thresh_low
        inds_high = scores < self.args.conf_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]

        cates = categories[remain_inds]
        cates_second = categories[inds_second]
        
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        """Step 1: Extract reid features"""
        if self.with_reid:
            features_keep = self.get_feature(tlwhs=dets[:, :4], ori_img=ori_img, crop_size=self.args.reid_crop_size)

        if len(dets) > 0:
            if self.with_reid:
                detections = [Tracklet_w_reid(tlwh, s, cate, motion=self.motion, feat=feat) for
                            (tlwh, s, cate, feat) in zip(dets, scores_keep, cates, features_keep)]
            else:
                detections = [Tracklet(tlwh, s, cate, motion=self.motion) for
                            (tlwh, s, cate) in zip(dets, scores_keep, cates)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_tracklets'''
        unconfirmed = []
        tracked_tracklets = []  # type: list[Tracklet]
        for track in self.tracked_tracklets:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracklets.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        tracklet_pool = BaseTracker.joint_tracklets(tracked_tracklets, self.lost_tracklets)

        # Predict the current location with Kalman
        for tracklet in tracklet_pool:
            tracklet.predict()

        dists = iou_distance(tracklet_pool, detections)

        # fuse detection conf into iou dist
        if self.args.fuse_detection_score:
            dists = fuse_det_score(dists, detections)

        if self.with_reid:
            # eq. 11 in Bot-SORT paper, i.e., the common method of 
            # fusing reid and motion. you can adjust the weight here
            emb_dists = embedding_distance(tracklet_pool, detections)
            dists = 0.9 * dists + 0.1 * emb_dists
        
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.9)

        for itracked, idet in matches:
            track = tracklet_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [Tracklet(tlwh, s, cate, motion=self.motion) for
                          (tlwh, s, cate) in zip(dets_second, scores_second, cates_second)]
        else:
            detections_second = []
        r_tracked_tracklets = [tracklet_pool[i] for i in u_track if tracklet_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked_tracklets, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_tracklets[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        for it in u_track:
            track = r_tracked_tracklets[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_tracklets.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        
        # fuse detection conf into iou dist
        if self.args.fuse_detection_score:
            dists = fuse_det_score(dists, detections)
       
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_tracklets.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracklets.append(track)

        """ Step 4: Init new tracklets"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.init_thresh:
                continue
            track.activate(self.frame_id)
            activated_tracklets.append(track)

        """ Step 5: Update state"""
        for track in self.lost_tracklets:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracklets.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_tracklets = [t for t in self.tracked_tracklets if t.state == TrackState.Tracked]
        self.merge_tracklets(activated_tracklets, refind_tracklets, lost_tracklets, removed_tracklets)

        output_tracklets = [track for track in self.tracked_tracklets if track.is_activated]

        return output_tracklets