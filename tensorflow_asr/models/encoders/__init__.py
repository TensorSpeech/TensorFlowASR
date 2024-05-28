import glob
from os.path import basename, dirname, isdir, isfile, join

for fd in glob.glob(join(dirname(__file__), "*")):
    if not isfile(fd) and not isdir(fd):
        continue
    if isfile(fd) and not fd.endswith(".py"):
        continue
    fd = fd if isdir(fd) else fd[:-3]
    fd = basename(fd)
    if fd.startswith("__"):
        continue
    __import__(f"{__name__}.{fd}")
