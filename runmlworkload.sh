apt-get update
apt install vim -y
apt install wget -y

while true
do echo "Running Pytorch Workload"
date
wget https://raw.githubusercontent.com/yogeshbendre/predictme/master/ysbclassifier.py
python3 ysbclassifier.py
echo "sleeping 2 minutes"
sleep 120
done
