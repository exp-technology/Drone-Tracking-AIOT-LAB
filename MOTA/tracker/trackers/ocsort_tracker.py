"""
OC Sort
"""

import numpy as np
from collections import deque
from .basetrack import BaseTrack, TrackState
from .tracklet import Tracklet, Tracklet_w_velocity
from .matching import *

# for reid
import torch
import torchvision.transforms as T
from .reid_models.engine import load_reid_model, crop_and_resize

# base class
from .basetracker import BaseTracker

class OCSortTracker(BaseTracker):
    def __init__(self, args, frame_rate=30):
        
        super().__init__(args, frame_rate=frame_rate)

        self.delta_t = 3

        # whether to use reid 
        self.with_reid = args.reid
        self.reid_model = None
        if self.with_reid:
            self.reid_model = load_reid_model(args.reid_model, args.reid_model_path, 
                                              device=args.device, trt=args.trt, crop_size=args.reid_crop_size)

        # once init, clear all trackid count to avoid large id
        BaseTrack.clear_count()

    @staticmethod
    def k_previous_obs(observations, cur_age, k):
        if len(observations) == 0:
            return [-1, -1, -1, -1, -1]
        for i in range(k):
            dt = k - i
            if cur_age - dt in observations:
                return observations[cur_age - dt]
        max_age = max(observations.keys())
        return observations[max_age]

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
            features_second = self.get_feature(tlwhs=dets_second[:, :4], ori_img=ori_img, crop_size=self.args.reid_crop_size)
            # in deep oc sort, low conf detections also need reid features

        if len(dets) > 0:
            '''Detections'''
            if self.with_reid:
                detections = [Tracklet_w_velocity(tlwh, s, cate, motion=self.motion, feat=feat) for
                              (tlwh, s, cate, feat) in zip(dets, scores_keep, cates, features_keep)]
            else:
                detections = [Tracklet_w_velocity(tlwh, s, cate, motion=self.motion) for
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

        ''' Step 2: First association, Observation Centric Momentum'''
        tracklet_pool = BaseTracker.joint_tracklets(tracked_tracklets, self.lost_tracklets)

        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in tracklet_pool])
        
        # last observation, obervation-centric
        # last_boxes = np.array([trk.last_observation for trk in tracklet_pool])

        # historical observations
        k_observations = np.array(
            [self.k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in tracklet_pool])


        # Predict the current location with Kalman
        for tracklet in tracklet_pool:
            tracklet.predict()

        # Observation centric cost matrix and assignment
        if self.with_reid:
            matches, u_track, u_detection = observation_centric_association_w_reid(
                tracklets=tracklet_pool, detections=detections, iou_threshold=0.3, 
                velocities=velocities, previous_obs=k_observations, vdc_weight=0.05
            )
        
        else:
            matches, u_track, u_detection = observation_centric_association(
                tracklets=tracklet_pool, detections=detections, iou_threshold=0.3, 
                velocities=velocities, previous_obs=k_observations, vdc_weight=0.05
            )

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
            if self.with_reid:
                detections_second = [Tracklet_w_velocity(tlwh, s, cate, motion=self.motion, feat=feat) for
                                  (tlwh, s, cate, feat) in zip(dets_second, scores_second, cates_second, features_second)]
            else:
                detections_second = [Tracklet_w_velocity(tlwh, s, cate, motion=self.motion) for
                            (tlwh, s, cate) in zip(dets_second, scores_second, cates_second)]
        else:
            detections_second = []

        r_tracked_tracklets = [tracklet_pool[i] for i in u_track if tracklet_pool[i].state == TrackState.Tracked]

        dists = 1. - iou_distance(r_tracked_tracklets, detections_second)
        if self.with_reid:  # for low confidence detections, we also use reid and add directly
            # note that embedding_distance calculate the 1. - cosine, not cosine
            emb_dists = 1. - embedding_distance(r_tracked_tracklets, detections_second, metric='cosine')
            dists = dists + emb_dists

        matches, u_track, u_detection_second = linear_assignment(-1 * dists, thresh=0.0)
        for itracked, idet in matches:
            track = r_tracked_tracklets[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        
        '''Step 4: Third association, match high-conf remain detections with last observation of tracks'''
        r_tracked_tracklets = [r_tracked_tracklets[i] for i in u_track]  # remain tracklets from last step
        r_detections = [detections[i] for i in u_detection]  # high-conf remain detections

        dists = 1. - ious(atlbrs=[t.last_observation[: 4] for t in r_tracked_tracklets],  # parse bbox directly
                          btlbrs=[d.tlbr for d in r_detections])

        matches, u_track, u_detection = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_tracklets[itracked]
            det = r_detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_tracklets.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_tracklets.append(track)

        # for tracks still failed, mark lost
        for it in u_track:
            track = r_tracked_tracklets[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_tracklets.append(track)        


        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [r_detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
       
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
    
    