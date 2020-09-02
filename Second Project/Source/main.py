#!/usr/bin/env python

import cv2
import numpy as np
import argparse
import math
import sys

from utils import *


if __name__ == '__main__':

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", type=int, required=True,
                    help="Which mode to use: 1 - Offside Player; 2 - Free kick; 3 - Distance to goal;")
    ap.add_argument("-i", "--image", required=True,
                    help="Source Image Path")
    args = vars(ap.parse_args())

    mode = args["mode"]
    img_src = args["image"]

    # Fetch blue points.
    img_dst = cv2.imread('plane_field.png')
    blue_points = get_blue_points(img_dst)
    pts_dst = np.array(blue_points, dtype=float)

    # Read source image.
    img_src = cv2.imread(img_src)
    size = img_src.shape

    # Get four corners of the field
    print('Select the homography points of the field and then press ENTER\nUsage: LMB to select a point, RMB to ignore if not found.')
    pts_src = get_field_points(img_src)

    # Remove invalid coordinates (due to ignoring points) from point lists.
    pts_src, pts_dst = get_homography_points(pts_src, pts_dst)

    # Get necessary point according to selected mode
    if mode == 1:
        print('Select the offside player and then press ENTER')
    elif mode == 2:
        print('Select the ball in order to create the free kick area and then press ENTER')
    elif mode == 3:
        print('Select a player and then press ENTER')
    player = get_point(img_src)

    # Get Goal if the Distance to Goal mode is selected
    if mode == 3:
        print('Select the middle of the goal line')
        goal = get_point(img_src)

    # Resize destination image in order to better fit in screen
    height = int(img_dst.shape[0] * 20 / 100)
    width = int(img_dst.shape[1] * 20 / 100)
    dim = (width, height)
    img_dst = cv2.resize(img_dst, dim, interpolation=cv2.INTER_AREA)

    # Calculate Homography between source and destination points
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to fit the plane one.
    im_temp = cv2.warpPerspective(
        img_src, h, (img_dst.shape[1], img_dst.shape[0]))

    # Black out polygonal area in destination image.
    cv2.fillConvexPoly(img_dst, np.array([[0, 0], [size[1], 0], [
                       size[1], size[0]], [0, size[0]]], dtype=float).astype(int), 0)

    # Add warped source image to destination image.
    img_dst = img_dst + im_temp

    # Convert player based on perspective transform.
    player_new = cv2.perspectiveTransform(np.array([player]), h)

    # Get player mask.
    player_mask = get_player_mask(img_src)

    # Close opened images
    cv2.destroyWindow("Image")
    cv2.destroyWindow("Field")

    # Prepare points for inverse homography
    if mode == 1:
        # Get edges of the field
        player_new = [[(player_new[0][0][0]), 0.0], [
            (player_new[0][0][0]), size[0]]]
    elif mode == 2:
        # Determine bounding box around point
        x = (player_new[0][0][0])
        y = (player_new[0][0][1])

        radius_px = (width * 9.15) / 110
        player_new = [[x - radius_px, y - radius_px], [x + radius_px, y - radius_px],
                      [x + radius_px, y + radius_px], [x - radius_px, y + radius_px]]
    elif mode == 3:
        # Calculate distance to goal
        x = (player_new[0][0][0])
        y = (player_new[0][0][1])

        player_new = [[x, y]]
        goal_new = cv2.perspectiveTransform(np.array([goal]), h)

        x2 = (goal_new[0][0][0])
        y2 = (goal_new[0][0][1])

        goal_new = [[x2, y2]]

        dist = math.sqrt((x-x2)**2 + (y-y2)**2)
        real_dist = dist * 110 / width

    # Calculate inverse Homography between source and destination points
    h_inv, status = cv2.findHomography(pts_dst, pts_src)

    # Convert player based on new, inverse perspective transform.
    player_best = cv2.perspectiveTransform(np.array([player_new]), h_inv)

    # Apply mask to get players in a new layer
    fg_back_inv = cv2.bitwise_and(img_src, img_src, mask=player_mask)

    # Draw necessary geometry
    if mode == 1:
        # Draw line in final image
        player_p1, player_p2 = player_best[0].astype(int)
        im_name = "Offside"

        img_src = cv2.line(
            img_src, (player_p1[0], player_p1[1]), (player_p2[0], player_p2[1]), (255, 0, 0), 2)

    elif mode == 2:
        # Draw ellipse in final image
        player_best = player_best[0].astype(int)
        box = cv2.minAreaRect(player_best)
        im_name = "Free kick"

        img_src = cv2.ellipse(img_src, box, (255, 0, 0), 2)

    elif mode == 3:
        # Draw distance to goal arrow in final image

        goal_best = cv2.perspectiveTransform(np.array([goal_new]), h_inv)

        player_p1 = player_best[0].astype(int)
        player_p2 = goal_best[0].astype(int)
        im_name = "Distance"

        img_dst = cv2.arrowedLine(img_src, (player_p1[0][0], player_p1[0][1]), (
            player_p2[0][0], player_p2[0][1]), (255, 0, 0), 2, cv2.LINE_AA, 0, 0.05)
        img_dst = cv2.arrowedLine(img_src, (player_p2[0][0], player_p2[0][1]), (
            player_p1[0][0], player_p1[0][1]), (255, 0, 0), 2, cv2.LINE_AA, 0, 0.05)
        cv2.putText(img_src, "{} meters".format(round(float(real_dist), 1)), (120, 120),
                    font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Remove applied player mask
    img_src = cv2.bitwise_or(img_src, fg_back_inv)

    # Show final image
    cv2.imshow(im_name, img_src)

    cv2.waitKey(0)
