import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import requests
import time
from emoji.unicode_codes import UNICODE_EMOJI
import emoji
from PIL import Image, ImageFont, ImageDraw

from avatar_class import AvatarProcessor

import main_translate
from main_translate import translate

from threading import Thread

capture = None

urlEmotions = "http://ds.ikorsakov.com:60001"

class CamHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header(
                'Content-type',
                'multipart/x-mixed-replace; boundary=--jpgboundary'
            )
            self.end_headers()
            counter = time.perf_counter()
            emoji = ""
            prev = counter

            avatar = AvatarProcessor("frozen_model.pb")
            prevTags = ""
            while True:
                try:

                    rc, img = capture.read()

                    if not rc:
                        continue


                    counter = time.perf_counter()
                    if ((counter - prev) > 5):
                        emoji = self.getEmotions(img)
                        prev = counter




                    #cv2.putText(img, "TES1111111111111111T", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    #            2, cv2.LINE_AA)
                    #img_str = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tostring()
                    cv2.putText(img, prevTags, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2, cv2.LINE_AA)

                    img = avatar.process_frame(img)
                    # draw Emoji
                    img = self.drawEmoji(img, emoji)

                    if (main_translate.globalTags != prevTags):
                        print(main_translate.globalTags)
                        print(' '.join([str(elem) for elem in main_translate.globalTags]))

                        prevTags = main_translate.globalTags

                    img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1]
                    img_str = img.tostring()

                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', len(img_str))
                    self.end_headers()

                    self.wfile.write(img_str)
                    self.wfile.write(b"\r\n--jpgboundary\r\n")


                except KeyboardInterrupt:
                    self.wfile.write(b"\r\n--jpgboundary--\r\n")
                    break
                except BrokenPipeError:
                    continue
            return

        if self.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><head></head><body>')
            self.wfile.write(b'<img src="http://127.0.0.1:8081/cam.mjpg"/>')
            self.wfile.write(b'</body></html>')
            return

    def drawEmoji(self, img, stringEmoji):
        if (stringEmoji == ""):
            return img
        im_p = Image.fromarray(img)
        # Get a drawing context
        draw = ImageDraw.Draw(im_p)
        font = ImageFont.truetype("c:\\Windows\\Fonts\\seguiemj.ttf", 72)
        tick = str(emoji.emojize(UNICODE_EMOJI[stringEmoji]))
        draw.text((5, 45), tick, (0, 192, 192), font=font)

        # Convert back to OpenCV image and save
        return np.array(im_p)

    def getEmotions(self, img):
        resultString = ""
        # send req to emotionServ
        reqEmotionsResponse = requests.post(urlEmotions, files={
            'frame': cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tostring()})
        dict = reqEmotionsResponse.json()[0]["emotions"]
        emotion = sorted(dict, key=dict.get)[-1]
        if (emotion == "angry"):
            resultString = u'\U0001f620'
        elif (emotion == "sad"):
            resultString = u'\U0001f622'
        elif (emotion == "happy"):
            resultString = u'\U0001f604'
        elif (emotion == "disgust"):
            resultString = u'\U0001f62C'
        elif (emotion == "neutral"):
            resultString = u'\U0001f610'
        elif (emotion == "fear"):
            resultString = u'\U0001f628'
        elif (emotion == "surprise"):
            resultString = u'\U0001f626'
        return resultString


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

def camera():
    global capture
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # thread.join()
    global img
    try:
        server = ThreadedHTTPServer(('localhost', 9000), CamHandler)
        print("server started at http://127.0.0.1:8081/cam.html")
        server.serve_forever()
    except KeyboardInterrupt:
        capture.release()
        server.socket.close()

def main():

    t1 = Thread(target=main_translate.translate, args=[])
    t1.start()
    camera()
    t1.join()

    #t1 = Thread(target=camera(), args=(1,), daemon=True)
    #t1.start()

    print("te")
    t2 = Thread(target=translate())
    t2.start()


    #thread1.start()
'''
    global capture
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    thread = Thread(target=main_translate.main())
    thread.start()
    thread.
    #thread.join()
    global img
    try:
        server = ThreadedHTTPServer(('localhost', 9000), CamHandler)
        print("server started at http://127.0.0.1:8081/cam.html")
        server.serve_forever()
    except KeyboardInterrupt:
        capture.release()
        server.socket.close()
'''



if __name__ == '__main__':
    main()