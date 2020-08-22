fstcompile --isymbols=chars.syms --acceptor Word.txt |
    fstcompose - ${1} |
    fstshortestpath |
    fstrmepsilon |
    fsttopsort |
    fstprint -osymbols=chars.syms |
    cut -f4 |
    grep -v "<epsilon>" |
    head -n -1 |
    tr -d '\n'
