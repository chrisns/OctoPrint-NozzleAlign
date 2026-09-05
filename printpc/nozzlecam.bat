@echo off
"C:\OctoPrint\ffmpeg.exe" -hide_banner -v error -f dshow -rtbufsize 200M -vcodec mjpeg -video_size 1280x800 -framerate 10 -i video="HD Camera" -c copy -f mjpeg -
