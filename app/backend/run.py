from subprocess import check_output
check_output("uvicorn main:app", shell=True)
