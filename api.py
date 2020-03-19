from __future__ import absolute_import

from asr.FlaskApi import app, socketio

if __name__ == '__main__':
  socketio.run(app, host='0.0.0.0')
