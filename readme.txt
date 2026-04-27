создал среду
python -m venv venv

Выбор среды 
 .\venv\Scripts\activate
.\venv\Scripts\Activate

 pip install opencv-python
 pip install pymavlink pyserial

 На Orange
 ///////////////////////////////////////
На Orange:
python -m venv venv
source ./venv/bin/activate

pip install -U pip
pip install opencv-contrib-python-headless pymavlink pyserial numpy


.\mavproxy.exe --master=udp:0.0.0.0:14550 --out=udpin:127.0.0.1:14551 --out=udpin:0.0.0.0:14552
.\mavproxy.exe --master=udp:0.0.0.0:14550 --out=udpin:127.0.0.1:14551 --out=udpin:0.0.0.0:14552
.\mavproxy.exe --master=udp:0.0.0.0:14550 --out=udp:127.0.0.1:14551 --out=udp:127.0.0.1:14552
.\mavproxy.exe --master=udp:0.0.0.0:14550 --out=udp:127.0.0.1:14551 --out=udpin:127.0.0.1:14552

 