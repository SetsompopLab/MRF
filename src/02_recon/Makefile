default:
	docker build -t setsompop/recon .

clean:
	rm -rf __pycache__
	rm -rf */__pycache__

save:
	docker build -t setsompop/recon .
	docker save setsompop/recon> recon.tar
