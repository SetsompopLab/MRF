default:
	docker build -t setsompop/calib .

clean:
	rm -rf __pycache__
	rm -rf */__pycache__

save:
	docker build -t setsompop/calib .
	docker save setsompop/calib> calib.tar
