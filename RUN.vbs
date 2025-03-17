Set WshShell = CreateObject("WScript.Shell")

strScriptFullPath = WScript.ScriptFullName

strScriptPath = Left(strScriptFullPath, Len(strScriptFullPath) - Len(WScript.ScriptName))

strBatFile = strScriptPath & "ASR_Start_test.bat"

WshShell.Run chr(34) & strBatFile & chr(34), 1

Set WshShell = Nothing