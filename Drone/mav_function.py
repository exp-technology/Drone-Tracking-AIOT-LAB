from pymavlink import mavutil


class DroneCommand:

    def __init__(self, drone_master=None):
        self.drone_master = drone_master

    def select_mode(self, mode):
        msg = mavutil.mavlink.MAVLink_command_long_message(0, 0, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                                                           1, mode, 0, 0, 0, 0, 0)
        self.drone_master.mav.send(msg)

    def arm_disarm(self, arm_status):
        msg = mavutil.mavlink.MAVLink_command_long_message(0, 0, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                                                           arm_status,  # 0:disarm 1:arm
                                                           0, 0, 0, 0, 0, 0)
        self.drone_master.mav.send(msg)

    def takeoff(self, altitude=0):
        msg = mavutil.mavlink.MAVLink_command_long_message(0, 0, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
                                                           0, 0, 0, 0, 0, 0, altitude)
        self.drone_master.mav.send(msg)

    def land(self, latitude=0, longitude=0):
        msg = mavutil.mavlink.MAVLink_command_long_message(0, 0, mavutil.mavlink.MAV_CMD_NAV_LAND, 0,
                                                           0, 0, 0, 0, latitude, longitude, 0)
        self.drone_master.mav.send(msg)

    def set_position_velocity(self, vx, vy, vz):
        msg = self.drone_master.mav.set_position_target_local_ned_encode(0, 0, 0,
                                                                        #  mavutil.mavlink.MAV_FRAME_LOCAL_NED, # MAV_FRAME_BODY_OFFSET_NED 
                                                                         mavutil.mavlink.MAV_FRAME_BODY_NED,
                                                                         0b10111000111,
                                                                         0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0)
        self.drone_master.mav.send(msg)

    def turning_yaw(self, heading, direction, angular_speed=0):
        msg = mavutil.mavlink.MAVLink_command_long_message(0, 0, mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0,
                                                           heading, angular_speed, direction,  # direction -1 ccw, 1 cw
                                                           1, 0, 0, 0)
        self.drone_master.mav.send(msg)

    def set_velocity_yawrate(self, vx, vy, vz, yaw_rate):
        msg = self.drone_master.mav.set_position_target_local_ned_encode(
            0, 0, 0,
            # mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            mavutil.mavlink.MAV_FRAME_BODY_NED, 
            0b10111000111,                         # 位置/加速度/ yaw 忽略，使用 velocity + yaw_rate
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, float(yaw_rate)                     # yaw 被 mask 忽略；yaw_rate 生效
        )
        self.drone_master.mav.send(msg)

    def servo(self, index, degree):
        msg = mavutil.mavlink.MAVLink_command_long_message(0, 0, mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0,
                                                           index, degree, 1, 0, 0, 0, 0)
        self.drone_master.mav.send(msg)

    # Receive data
    def read_data_command(self, command_id, times=500000):
        message = self.drone_master.mav.command_long_encode(self.drone_master.target_system,
                                                            self.drone_master.target_component,
                                                            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                                                            0, command_id, times, 0, 0, 0, 0, 0)
        self.drone_master.mav.send(message)

