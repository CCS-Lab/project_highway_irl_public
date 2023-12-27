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

session_num = 'prac'
duration = 5

run_highway(sub_num,session_num,duration,0)
 
ctypes.windll.user32.MessageBoxW(0, "연습이 끝났습니다. 본 실험을 하실 준비가 되었다면 확인 버튼을 눌러주세요.", "연습 종료", 0)

session_num = '01'
duration = 20

run_highway(sub_num,session_num,duration,1)
ctypes.windll.user32.MessageBoxW(0, "잠시 휴식을 취하실 수 있는 시간입니다. 다시 실험을 재개하고 싶으실 때 확인 버튼을 눌러주세요.", "휴식 시간", 0)

session_num = '02'

run_highway(sub_num,session_num,duration,1)
ctypes.windll.user32.MessageBoxW(0, "실험이 종료되었습니다. 실험 담당자에게 알려주세요.", "실험 종료", 0)