from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame, pygame.image
from pygame.locals import *

import numpy as np
import math
import re
import glob
import cv2
import imageio
# import matplotlib.pyplot as plt


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


'''
OpenGL
GL_PROJECTION: camaera intrinsics
GL_MODELVIEW: relative pose between object and camera
'''

def set_projection_from_camera(K, im_w, im_h):
  """  Set view from a camera calibration matrix.
  K: calibration matrix
  """

  glMatrixMode(GL_PROJECTION)
  # glLoadMatrixf(K)
  glLoadIdentity()

  fx = K[0,0]
  fy = K[1,1]
  fovy = 2*math.atan(0.5*im_h/fy)*180/math.pi  # arctan
  aspect = (im_w*fy)/(im_h*fx)

  # define the near and far clipping planes
  near = 0.00001
  far = 10000.0

  # set perspective
  gluPerspective(fovy,aspect,near,far)
  glViewport(0,0,im_w,im_h)


def set_modelview_from_camera(Rt):
  """  Set the model view matrix from camera pose. """

  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()

  # rotate teapot 90 deg around x-axis so that z-axis is up
  Rx = np.array([[1,0,0],[0,0,-1],[0,1,0]])

  # set rotation to best approximation
  R = Rt[:,:3]
  U,S,V = np.linalg.svd(R)
  R = U @ V
  R[0,:] = -R[0,:] # change sign of x-axis

  # set translation
  t = Rt[:,3].reshape(-1)

  # setup 4*4 model view matrix
  M = np.eye(4)
  M[:3,:3] = R @ Rx
  M[:3,3] = t

  # transpose and flatten to get column order
  M = M.T
  m = M.flatten()

  # replace model view with the new matrix
  glLoadMatrixf(m)


def draw_background(imname, im_w, im_h):
  """  Draw background image using a quad. """

  # load background image (should be .bmp) to OpenGL texture
  bg_image = pygame.image.load(imname).convert()
  bg_data = pygame.image.tostring(bg_image,"RGBX",1)

  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

  # bind the texture
  glEnable(GL_TEXTURE_2D)
  glBindTexture(GL_TEXTURE_2D,glGenTextures(1))
  glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,im_w,im_h,0,GL_RGBA,GL_UNSIGNED_BYTE,bg_data)
  glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
  glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)

  # create quad to fill the whole window
  glBegin(GL_QUADS)
  glTexCoord2f(0.0,0.0); glVertex3f(-1.0,-1.0,-1.0)
  glTexCoord2f(1.0,0.0); glVertex3f( 1.0,-1.0,-1.0)
  glTexCoord2f(1.0,1.0); glVertex3f( 1.0, 1.0,-1.0)
  glTexCoord2f(0.0,1.0); glVertex3f(-1.0, 1.0,-1.0)
  glEnd()

  # clear the texture
  glDeleteTextures(1)




def load_and_draw_model(filename, scale=1.0):
  """  Loads a model from an .obj file using objloader.py.
    Assumes there is a .mtl material file with the same name. """
  glEnable(GL_LIGHTING)
  glEnable(GL_LIGHT0)
  glEnable(GL_DEPTH_TEST)
  glClear(GL_DEPTH_BUFFER_BIT)

  # set model color
  glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])

  # glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.75,1.0,0.0])
  glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.863,  0.569,  0.118])
  # glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.2, 0.1, 0.01])

  glMaterialf(GL_FRONT,GL_SHININESS, 1)
  # glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0)

  # load from a file
  import objloader
  obj = objloader.OBJ(filename,swapyz=True)
  if scale != 1:
    glScalef(scale, scale, scale);
  glCallList(obj.gl_list)



def setup(im_w, im_h):
  """ Setup window and pygame environment. """
  pygame.init()
  window = pygame.display.set_mode((im_w, im_h),OPENGL | DOUBLEBUF)
  pygame.display.set_caption('OpenGL AR demo')
  return window





if __name__ == '__main__':
    # read file
    BLACK_BG = True
    BLEND_DIRECTRLY = True


    scene_name = 'room'  #  bridge desk rgbd_dataset_freiburg1_desk
    # read file
    bg_filenames = glob.glob('../Data/%s/texture/*.bmp' %scene_name)
    bg_filenames.sort()
    has_pose, poses = read_kerframe_trajectory('%s-KeyFrameTrajectory.txt'%scene_name, '../Data/%s/rgb.txt'%scene_name)
    # has_pose, poses = read_trajectory('%s-AllTrajectory.txt'%scene_name)


    # bg_filenames = glob.glob('../Data/desk3/texture/*.bmp')
    # bg_filenames.sort()
    # has_pose, poses = read_trajectory('desk-AllTrajectory.txt')
    # has_pose, poses = read_kerframe_trajectory('desk-KeyFrameTrajectory.txt', '../Data/desk/rgb.txt')


    bg_filenames = np.array(bg_filenames)
    bg_filenames = bg_filenames[has_pose]
    assert(len(poses) == len(bg_filenames))

    n_frames = len(poses)

    # Camera Intrinsics
    im_w, im_h = 1280, 720
    K =  np.array([ [1255.9,  0, 640],
                    [0, 1262.28, 360],
                    [0,       0,   1]])
    window = setup(im_w, im_h)

    result_imgs = []
    for i in range(n_frames):
      print(i)
      # Extrinsics
      # model pose w.r.t first camera
      # t[0]=right,  t[1]=up, t[3]= -depth
      t_model = np.array([[10, 0, -100]]).T   # fox
      # t_model = np.array([[4, 5, -150]]).T   # plane
      R_model = degree2R(roll=0, pitch=0, yaw=0)
      Rt_model = np.hstack([R_model, t_model])  # [3,4]
      Rt_model = np.vstack([Rt_model, [0,0,0,1]])  #[4,4]
      # Rt_model =  np.array([  [ 1, 0, 0,    4],
      #                         [ 0, 1, 0,    5],
      #                         [ 0, 0, 1, -150],
      #                         [ 0, 0, 0,   1]])

      # Camera Pose
      Rt_cam = poses[i]
      Rt_model = Rt_cam @ Rt_model
      # Rt_model = np.dot(Rt_model, Rt_cam)

      Rt = Rt_model[:-1, :]  # [4,4] -> [3,4]

      # setup(im_w, im_h)
      if BLACK_BG:
        draw_background('blackbg.bmp', im_w, im_h)
      else:
        draw_background(bg_filenames[i], im_w, im_h)
      set_projection_from_camera(K, im_w, im_h)
      set_modelview_from_camera(Rt)

      load_and_draw_model('./fox.obj', scale=.1)


      outdir = 'tmp'


      pygame.display.flip()
      pygame.image.save(window, '%s/%d.png' % (outdir, i))
      if BLEND_DIRECTRLY:
        img_bg = cv2.imread(bg_filenames[i])
        img_front = cv2.imread('%s/%d.png' % (outdir, i))
        mask = (img_front.sum(-1) == 0)
        img_blended = img_front.copy()
        img_blended[mask] = img_bg[mask]
        cv2.imwrite('%s/%d.png' % (outdir, i), img_blended)

        cv2.imshow('blended', img_blended)
        result_imgs.append(img_blended[..., ::-1])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      # input('Press to continue...')
    imageio.mimsave('result/%s-3d.gif'%scene_name, result_imgs)

      # while True:
      #   event = pygame.event.poll()
      #   if event.type in (QUIT,KEYDOWN):
      #     break
      #   pygame.display.flip()

