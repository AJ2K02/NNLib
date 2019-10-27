path = input("Path : ")
outpath = 'out_' + path
file = open(path, 'r')
out = open(outpath, 'w')
content = file.read().split('\n')
for line in content:
	if line.strip(' \t').startswith("<font size=\"4\">"):
		out.write('<hr>\n\n')
	out.write(line)
