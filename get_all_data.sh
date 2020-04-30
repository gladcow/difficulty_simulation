python3 ./test_mining.py -a dgw3 -s const >> results.txt 2>>results.txt
python3 ./test_mining.py -a xmr -s const >> results.txt 2>>results.txt
python3 ./test_mining.py -a sa -s const >> results.txt 2>>results.txt
python3 ./test_mining.py -a lwma -s const >> results.txt 2>>results.txt

python3 ./test_mining.py -a dgw3 -s random >> results.txt 2>>results.txt
python3 ./test_mining.py -a xmr -s random >> results.txt 2>>results.txt
python3 ./test_mining.py -a sa -s random >> results.txt 2>>results.txt
python3 ./test_mining.py -a lwma -s random >> results.txt 2>>results.txt

python3 ./test_mining.py -a dgw3 -s increase >> results.txt 2>>results.txt
python3 ./test_mining.py -a xmr -s increase >> results.txt 2>>results.txt
python3 ./test_mining.py -a sa -s increase >> results.txt 2>>results.txt
python3 ./test_mining.py -a lwma -s increase >> results.txt 2>>results.txt

python3 ./test_mining.py -a dgw3 -s decrease >> results.txt 2>>results.txt
python3 ./test_mining.py -a xmr -s decrease >> results.txt 2>>results.txt
python3 ./test_mining.py -a sa -s decrease >> results.txt 2>>results.txt
python3 ./test_mining.py -a lwma -s decrease >> results.txt 2>>results.txt

python3 ./test_mining.py -a dgw3 -s inout >> results.txt 2>>results.txt
python3 ./test_mining.py -a xmr -s inout >> results.txt 2>>results.txt
python3 ./test_mining.py -a sa -s inout >> results.txt 2>>results.txt
python3 ./test_mining.py -a lwma -s inout >> results.txt 2>>results.txt
