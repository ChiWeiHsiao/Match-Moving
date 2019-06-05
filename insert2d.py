import numpy as np
import math
import re
import glob
import argparse
import cv2
import os
import imageio
from objloader_simple import *


def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion
    Parameters
    ----------
    q : 4 element array-like
    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*
    '''
    _FLOAT_EPS = np.finfo(np.float).eps
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < _FLOAT_EPS:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])


def read_kerframe_trajectory(trajectory_fn, timestamp_fn):
    f = open(trajectory_fn, 'r')  # 'KeyFrameTrajectory.txt'
    lines = f.readlines()
    f.close()
    time_f = open(timestamp_fn, 'r')
    time_lines = time_f.readlines()
    time_f.close()

    Rt_list = []
    timestamp_list = []
    for line in lines:
      if line.strip():
        nums = [float(n) for n in line.split(' ')]
        timestamp = nums[0]
        t = np.array(nums[1:4]).reshape([3,1])
        R = quat2mat(nums[4:])
        Rt = np.hstack([R, t])
        Rt = np.vstack([Rt, [0,0,0,1]])
        Rt_list.append(Rt)
        timestamp_list.append(timestamp)


    has_pose = []
    for line in time_lines:
      if line.strip() and line[0] != '#':
        tstamp, imgpth = line.split(' ')
        if float(tstamp) in timestamp_list:
          has_pose.append(True)
        else:
          has_pose.append(False)


    Rt_list = np.array(Rt_list)
    has_pose = np.array(has_pose)

    return has_pose, Rt_list



def read_trajectory(filename):
    f = open(filename, 'r')
    # f = open('desk2-AllTrajectory.txt', 'r')

    lines = f.readlines()
    f.close()

    has_pose = []
    matrix_list = []
    i = 0
    while i < len(lines)-1:
        if lines[i] in ['\n', '\r\n']:
            i += 1
            break

        row = re.sub(r'\[|\]|\n|,|;', '', lines[i]).split()
        i += 1

        if len(row) == 0:
            has_pose.append(False)
        else:
            has_pose.append(True)
            matrix = []
            matrix.append([float(v) for v in row])
            for _ in range(3):
                row = re.sub(r'\[|\]|\n|,|;', '', lines[i]).split()
                matrix.append([float(v) for v in row])
                i += 1
            assert(len(matrix) == 4)
            matrix_list.append(matrix)
    matrix_list = np.array(matrix_list)
    has_pose = np.array(has_pose)

    return has_pose, matrix_list





def degree2R(roll, pitch, yaw):
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)

    yawMatrix = np.matrix([
    [math.cos(yaw), -math.sin(yaw), 0],
    [math.sin(yaw), math.cos(yaw), 0],
    [0, 0, 1]
    ])

    pitchMatrix = np.matrix([
    [math.cos(pitch), 0, math.sin(pitch)],
    [0, 1, 0],
    [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    rollMatrix = np.matrix([
    [1, 0, 0],
    [0, math.cos(roll), -math.sin(roll)],
    [0, math.sin(roll), math.cos(roll)]
    ])

    R = yawMatrix * pitchMatrix * rollMatrix

    return R




def approx_rotation(Rt):
  """  Get legal rotation matrix """
  # rotate teapot 90 deg around x-axis so that z-axis is up
  # Rx = np.array([[1,0,0],[0,0,-1],[0,1,0]])

  # set rotation to best approximation
  R = Rt[:3,:3]
  U,S,V = np.linalg.svd(R)
  R = np.dot(U,V)
  # R[0,:] = -R[0,:] # change sign of x-axis

  # set translation
  t = Rt[:3, 3].reshape(-1)

  # setup 4*4 model view matrix
  M = np.eye(4)
  # M[:3,:3] = np.dot(R,Rx)
  M[:3, :3] = R
  M[:3, 3] = t
  return M




if __name__ == '__main__':

    scene_name = 'room'  # bridge desk  rgbd_dataset_freiburg1_desk

    # read file
    bg_filenames = glob.glob('../Data/%s/rgb/*.png' %scene_name)
    bg_filenames.sort()
    has_pose, poses = read_kerframe_trajectory('%s-KeyFrameTrajectory.txt'%scene_name, '../Data/%s/rgb.txt'%scene_name)
    # has_pose, poses = read_trajectory('%s-AllTrajectory.txt'%scene_name)

    # # read file
    # bg_filenames = glob.glob('../Data/desk/rgb/*.png')
    # bg_filenames.sort()
    # # has_pose, poses = read_trajectory('../ORB_SLAM2/AllTrajectory.txt')
    # has_pose, poses = read_kerframe_trajectory('KeyFrameTrajectory.txt', '../Data/rgbd_dataset_freiburg1_desk/rgb.txt')

    print(len(bg_filenames), len(has_pose))
    bg_filenames = np.array(bg_filenames)
    bg_filenames = bg_filenames[has_pose]

    print(len(poses), len(bg_filenames))
    assert(len(poses) == len(bg_filenames))

    n_frames = len(poses)

    obj = cv2.imread('rirakuma.png', cv2.IMREAD_UNCHANGED)
    obj_2d = np.array([
      [0, 0],
      [0, obj.shape[0]],
      [obj.shape[1], obj.shape[0]],
      [obj.shape[1], 0],
    ], np.float32)

    # Camera Intrinsics
    # im_w, im_h = 1280, 720
    # K =  np.array([ [1255.9,  0, 640],
    #                 [0, 1262.28, 360],
    #                 [0,       0,   1]])
    im_w, im_h = 640, 480
    K =  np.array([ [517.306408,  0, 318.643040],
                    [0, 516.469215, 255.313989],
                    [0,       0,   1]])

    # Convert K [3,3] to [4,4]
    K = np.hstack([K, np.zeros([3,1])])
    K = np.vstack([K, [0,0,0,1]])

    result_imgs = []
    for i in range(n_frames):
      print('='*10)
      print('Frame: ', i)

      img = cv2.imread(bg_filenames[i])

      # Extrinsics
      # model pose w.r.t first camera
      # t_model = np.array([[250, 250, 3]]).T
      t_model = np.array([[0, 0, 5000]]).T
      R_model = degree2R(roll=0, pitch=0, yaw=180)
      Rt_model = np.hstack([R_model, t_model])  # [3,4]
      Rt_model = np.vstack([Rt_model, [0,0,0,1]])  #[4,4]

      # # Camera Pose
      Rt_cam = poses[i]
      # # Rt_cam = np.linalg.inv(Rt_cam)

      # Rt = Rt_cam @ Rt_model
      print(Rt_model)
      print(Rt_cam)
      # print('Rt', Rt, end='\n\n')
      P = K @ np.linalg.inv(Rt_cam) @ Rt_model
      # print('P', P, end='\n\n')
      # P = P[:-1, :]  # [4,4] -> [3,4]

      # Compute target image coordinates
      obj_3d = np.array([
        [-obj.shape[1]/2, -obj.shape[0]/2, 0, 1],
        [-obj.shape[1]/2, obj.shape[0]/2, 0, 1],
        [obj.shape[1]/2, obj.shape[0]/2, 0, 1],
        [obj.shape[1]/2, -obj.shape[0]/2, 0, 1],
      ], np.float32)
      cam_3d = (P @ obj_3d.T).T
      cam_2d = cam_3d[:, :2] / cam_3d[:, [2]]

      # Warp image
      M = cv2.getPerspectiveTransform(obj_2d.astype(np.float32), cam_2d.astype(np.float32))
      warp = cv2.warpPerspective(obj, M, (img.shape[1], img.shape[0]))

      # Blending
      alpha = warp[..., [3]] / 255.0
      blend = (warp[..., :3] * alpha + img * (1 - alpha)).astype(np.uint8)


      cv2.imshow('ImageWindow', blend)
      result_imgs.append(blend[..., ::-1])

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

      # input('Press to Continue...')
    imageio.mimsave('result/%s-2d.gif'%scene_name, result_imgs)
