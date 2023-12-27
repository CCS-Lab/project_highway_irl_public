from run_highway import run_highway
import ctypes  # An included library with Python install.
import wx

def ask(parent=None, message='', default_value=''):
    dlg = wx.TextEntryDialog(parent, message, value=default_value)
    dlg.ShowModal()
    result = dlg.GetValue()
    dlg.Destroy()
    return result
##Initialize wx App
app = wx.App()
sub_num = ask(message = 'Enter participant ID')

if sub_num !='':
    session_num = 'prac'
    duration_prac = 5

    run_highway(sub_num,session_num,duration_prac,0)

    ctypes.windll.user32.MessageBoxW(0, "Practice is over. Press OK to continue.", "Practice over", 0)

    session_num = '01'
    duration = 17

    run_highway(sub_num,session_num,duration,1)
    ctypes.windll.user32.MessageBoxW(0, "You may take a break. Press OK to continue.", "Break", 0)

    session_num = '02'

    run_highway(sub_num,session_num,duration,1)
    ctypes.windll.user32.MessageBoxW(0, "You have completed the task. Thank you.", "Finished", 0)