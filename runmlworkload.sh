apt-get update
apt install vim -y
apt install wget -y
wget https://raw.githubusercontent.com/yogeshbendre/predictme/master/ysbclassifier.py
while true
do echo "Running Pytorch Workload"
date
python3 ysbclassifier.py
echo "sleeping 2 minutes"
sleep 120
done
