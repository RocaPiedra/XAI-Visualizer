
import cv2
import numpy as np
import win32gui, win32ui, win32con
import numpy as np
import pyautogui


import numpy as np
import win32gui, win32ui, win32con


class WindowCapture:

    # properties
    w = 0
    h = 0
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self, class_name = None, window_name = None):
        # find the handle for the window we want to capture
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))

        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        # account for the window border and titlebar and cut them off
        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

        # set the cropped coordinates offset so we can translate screenshot
        # images into actual screen positions
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

    def get_screenshot(self):

        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        #dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type() 
        #   && _img.dims() <= 2 in function 'cv::matchTemplate'
        img = img[...,:3]

        # make image C_CONTIGUOUS to avoid errors that look like:
        #   File ... in draw_rectangles
        #   TypeError: an integer is required (got type tuple)
        # see the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
        img = np.ascontiguousarray(img)

        return img

    # find the name of the window you're interested in.
    # once you have it, update window_capture()
    # https://stackoverflow.com/questions/55547940/how-to-get-a-list-of-the-name-of-every-open-window
    def list_window_names(self):
        window_list = []
        hex_list = []
        list = []
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hwnd, hex(hwnd), win32gui.GetWindowText(hwnd))
                window_list.append(win32gui.GetWindowText(hwnd))
                hex_list.append(hex(hwnd))
                list.append(hwnd)
        win32gui.EnumWindows(winEnumHandler, None)
        return window_list, hex_list, list

    # translate a pixel position on a screenshot image to a pixel position on the screen.
    # pos = (x, y)
    # WARNING: if you move the window being captured after execution is started, this will
    # return incorrect coordinates, because the window position is only calculated in
    # the __init__ constructor.
    def get_screen_position(self, pos):
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)

    def capture_window(self):
        while True:
            window = WindowCapture.get_screenshot(self)
            
            cv2.imshow("window",window)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

    def capture_pyautogui():
        while True:
            window_image = pyautogui.screenshot()
            window = cv2.cvtColor(np.array(window_image), cv2.COLOR_RGB2BGR)
            #window = window[:,:,::-1]
            cv2.imshow("window",window)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

    def find_window(self):
        initial_list = []
        final_list = []
        new_window = []
        _,_,initial_list = self.list_window_names()
        input("Press enter when the window is loaded")
        _,_,final_list = self.list_window_names()
        # new_window = set(final_list)-set(initial_list)

        # if set is empty
        if new_window == set():
            print("No new windows opened")
            return None
        print("The new windows are:")
        for item in new_window:
            print(item)
        listed = list(new_window)
        return listed


def capture_cam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("camera failed to open")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("frame is not received from camera")
            break
        cv2.imshow("webcam",ret)
        
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

    cap.release()




if __name__ == '__main__':
    
    list = []
    wSearch = WindowCapture()
    list = wSearch.list_window_names()

    print('\n\n\n')
    i = 0
    for items in list[2]:
        i += 1
        print('{}: {}'.format(i, items))

    j = 0
    for items in list:
        j += 1
        print('{}: {}'.format(i, items))

    print('\n\n\n')
    
    new_windows = wSearch.find_window()
    print("New windows opened: ",new_windows)

    if new_windows != None:
        name_list = new_windows[0]
        hex_list = new_windows[1]
        id_list = new_windows[2]
        for i, items in id_list:
            print("{}: {}".format(i,items))
    else:
        print("There are no windows to capture")
        exit()
    
    print('\n\n\n')

    class_id = input("choose window to capture \n")
    wCap = WindowCapture(class_id, None)
    wCap.capture_window()
    


