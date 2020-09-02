import cv2
import numpy as np

INDEX = 1
MAX_POINTS = 12
font = cv2.FONT_HERSHEY_SIMPLEX


# Print instructions for point selection
def print_instructions():
    global INDEX

    if (INDEX <= MAX_POINTS):
        print(f'[{INDEX}]:', end=' ', flush=True)
        INDEX = INDEX + 1
    else:
        print('All points registered. Please press ENTER.')


# Handle left and right mouse click for point selection
def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(data['im'], (x, y), 3, (0, 0, 255), 5, 16)
        cv2.imshow("Image", data['im'])
        data['points'].append([x, y])

        print(f'Selected. Positioned at ({x}, {y}).')
        print_instructions()

    if event == cv2.EVENT_RBUTTONDOWN:
        data['points'].append([-1, -1])

        print('Ignored.')
        print_instructions()


# Load selected points into an array
def get_field_points(im):
    # Set up data to send to mouse handler
    data = {}

    data['im'] = im.copy()
    data['points'] = []

    # Set the callback function for any mouse event
    cv2.imshow("Image", im)

    print_instructions()

    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)

    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)

    return points


# Load selected single point into an array
def get_point(im):

    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []

    # Set the callback function for any mouse event
    cv2.imshow("Image", im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)

    # Convert array to np.array
    point = np.vstack(data['points']).astype(float)

    return point


# Remove invalid coordinates (due to ignoring points) from point lists.
def get_homography_points(pts_src, pts_dst):
    removable = []

    for i, pt_src in enumerate(pts_src):
        if all(a == -1 for a in pt_src):
            removable.append(i)

    pts_src = [pt_src for pt_src in pts_src if any(a != -1 for a in pt_src)]
    pts_src = np.vstack(pts_src).astype(float)

    pts_dst = np.delete(pts_dst, removable, 0)

    return pts_src, pts_dst


# Create a mask that only selects the players in the field
def get_player_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 20, 100), (70, 255, 255))
    return cv2.bitwise_not(mask)


# Get points for homography in plane image
def get_blue_points(img):
    scale_percent = 20
    height = int(img.shape[0] * scale_percent / 100)
    width = int(img.shape[1] * scale_percent / 100)
    dim = (width, height)
    index = 1

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Masks out any green color
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue = cv2.bitwise_and(img, img, mask=mask)

    blue_points = []

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        blue_points.append([int(round(x + w/2)), int(round(y + h/2))])

        cv2.putText(img, str(index), (int(round(x + w/2)), int(round(y + h/2))),
                    font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(blue, (x, y), (x + w, y + h), (0, 0, 255), 1)
        index += 1

    cv2.imshow("Field", img)

    return blue_points
