' Silent AegisRAG launcher — no console window
Option Explicit
Dim sh, fso, root, cmd
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
cmd = """" & root & "\venv\Scripts\pythonw.exe"" """ & root & "\run.py"""
sh.CurrentDirectory = root
sh.Run cmd, 0, False
