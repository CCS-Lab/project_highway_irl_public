from psychopy import gui
from run_highway import run_highway
import ctypes  # An included library with Python install.

expName = 'Highway_practice'
expInfo = {'Participant':'', 'session':'prac'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False,title=expName)
if dlg.OK == False:
    core.quit()
sub_num = expInfo['Participant']
session_num = expInfo['session']

duration = 5

run_highway(sub_num,session_num,duration,0)