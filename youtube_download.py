import youtube_dl
import os
import shutil

ydl_opts = {
    'format': 'bestaudio/mp3',
    'outtmpl': 'temp\\temp.webm',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '256',
        
    }],
}
def remove(value, deletechars):
    for c in deletechars:
        value = value.replace(c,'')
    return value
def get_audio(link):
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        # ydl.download([link])
        cwd = os.getcwd()
        name = ydl.extract_info(link, download=True).get('title', None)
        name = remove(name, '\/:*?"<>|')
        shutil.move(cwd + '\\temp\\temp.wav', cwd + '\\temp\\'+ name +'.wav')
        # os.rename(cwd + '\\temp\\temp.wav', cwd + '\\temp\\'+ name +'.wav')
        return name
