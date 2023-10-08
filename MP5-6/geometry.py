# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
import math
from alien import Alien
from typing import List, Tuple
from copy import deepcopy

def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    # Get the line segment that represents the alien's current shape
    head, tail = alien.get_head_and_tail()
    centroid = alien.get_centroid()
    for wall in walls:
        x1,y1,x2,y2 = wall[0], wall[1],wall[2],wall[3]
        tuple_wall = ((x1,y1),(x2,y2))
        if alien.get_shape() == 'Ball':
            #determine the distance of centroid to the line
            distance = point_segment_distance(centroid, tuple_wall)
            if distance <= alien.get_width():
                return True
        if alien.get_shape() == 'Horizontal':
            #determine the line segment intersect with the wall
            distance = segment_distance(tuple_wall, (head,tail))
            if  distance <= alien.get_width():
                return True
            
        if alien.get_shape() == 'Vertical':
            #determine the line segment intersect with the wall
            distance = segment_distance(tuple_wall, (head,tail))
            if  distance <= alien.get_width():
                return True
        

    return False


def is_alien_within_window(alien: Alien, window: Tuple[int, int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    centroid = alien.get_centroid()
    radius = alien.get_width()
    width, height = window

    if alien.get_shape() == 'Ball':
        if centroid[0] - radius <= 0 or centroid[0] + radius >= width or \
                centroid[1] - radius <= 0 or centroid[1] + radius >= height:
            return False
    elif alien.get_shape() == 'Horizontal':
        head, tail = alien.get_head_and_tail()
        if tail[0] - radius <= 0  or head[0] + radius >= width or \
                tail[1] - radius <= 0 or tail[1] + radius >= height:
            return False
    elif alien.get_shape() == 'Vertical':
        head, tail = alien.get_head_and_tail()
        if head[1] - radius <= 0  or tail[1] + radius >= height or \
                tail[0] - radius <= 0 or tail[0] + radius >= width:
            return False
        
    return True


def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    x, y = point
    n = len(polygon)

    res = []
    for i in range(n):
        # Calculate the cross product of vectors
        cross_product = (polygon[(i+1)%n][0] - polygon[i%n][0]) * (y - polygon[i%n][1]) - \
                        (polygon[(i+1)%n][1] - polygon[i%n][1]) * (x - polygon[i%n][0])
        res.append(cross_product)

    min_x = min(polygon, key=lambda x:x[0])[0]
    max_x = max(polygon, key=lambda x:x[0])[0]
    min_y = min(polygon, key=lambda x:x[1])[1]
    max_y = max(polygon, key=lambda x:x[1])[1]

    if min_x <= x <= max_x and min_y <= y <= max_y:
        if all([i>=0 for i in res]) or all([i<=0 for i in res]):
            return True
    
    return False


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    # Get the current position of the alien
    current_pos = alien.get_centroid()
    current_head, current_tail = alien.get_head_and_tail()

    target_alien = deepcopy(alien)
    target_alien.set_alien_pos(waypoint)
    target_head, target_tail = target_alien.get_head_and_tail()

    # Check if the line intersects any of the walls
    for wall in walls:
        x1,y1,x2,y2 = wall[0], wall[1],wall[2],wall[3]
        tuple_wall = ((x1,y1),(x2,y2))
        if segment_distance((current_pos, waypoint), tuple_wall) <= alien.get_width() or \
            segment_distance((current_head, target_head), tuple_wall) <= alien.get_width() or \
                segment_distance((current_tail, target_tail), tuple_wall) <= alien.get_width():
            return True
        
    # Check if the updated alien touches a wall
    if does_alien_touch_wall(target_alien, walls):
        return True

    return False


def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    x, y = p
    x1, y1 = s[0]
    x2, y2 = s[1]

    # Calculate the length of the segment
    segment_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # If the segment has zero length, return the distance to the first endpoint
    if segment_length == 0:
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)

    # Calculate the dot product of the vector from the first endpoint to the point and the vector from the first endpoint to the second endpoint
    dot_product = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / segment_length ** 2

    # Calculate the closest point on the segment to the point
    closest_x = x1 + dot_product * (x2 - x1)
    closest_y = y1 + dot_product * (y2 - y1)

    # If the closest point is outside the segment, return the distance to the closest endpoint
    if dot_product < 0:
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    elif dot_product > 1:
        return math.sqrt((x - x2) ** 2 + (y - y2) ** 2)

    # Calculate the distance from the point to the closest point on the segment
    return math.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)


def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    # Unpack the coordinates of the endpoints of the segments
    x1, y1 = s1[0]
    x2, y2 = s1[1]
    x3, y3 = s2[0]
    x4, y4 = s2[1]

    # Calculate the slopes and y-intercepts of the lines containing the segments
    if x2 - x1 == 0:
        slope1 = float('inf')
        y_intercept1 = x1
    else:
        slope1 = (y2 - y1) / (x2 - x1)
        y_intercept1 = y1 - slope1 * x1

    if x4 - x3 == 0:
        slope2 = float('inf')
        y_intercept2 = x3
    else:
        slope2 = (y4 - y3) / (x4 - x3)
        y_intercept2 = y3 - slope2 * x3

    # Check if the segments are parallel
    if slope1 == slope2:
        # on the same line
        if y_intercept1 == y_intercept2:
            if max(x1, x2) < min(x3, x4) or max(x3, x4) < min(x1, x2):
                return False
            else:
                return True
        else:
            return False

    # Calculate the intersection point of the lines containing the segments
    if slope1 == float('inf'):
        x_intersect = x1
        y_intersect = slope2 * x_intersect + y_intercept2
    elif slope2 == float('inf'):
        x_intersect = x3
        y_intersect = slope1 * x_intersect + y_intercept1
    else:
        x_intersect = (y_intercept2 - y_intercept1) / (slope1 - slope2)
        y_intersect = slope1 * x_intersect + y_intercept1

    # Check if the intersection point is within both segments
    if (min(x1, x2) <= x_intersect <= max(x1, x2) and
            min(y1, y2) <= y_intersect <= max(y1, y2) and
            min(x3, x4) <= x_intersect <= max(x3, x4) and
            min(y3, y4) <= y_intersect <= max(y3, y4)):
        return True

    return False


def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    # Check if the segments intersect
    if do_segments_intersect(s1, s2):
        return 0.0

    # Calculate the distances from the endpoints of segment1 to segment2
    distances = []
    for p in s1:
        distances.append(point_segment_distance(p, s2))

    # Calculate the distances from the endpoints of segment2 to segment1
    for p in s2:
        distances.append(point_segment_distance(p, s1))

    # Return the minimum distance
    return min(distances)


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i][j][k]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config}' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            
    # Initialize Aliens and perform simple sanity check.
    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
