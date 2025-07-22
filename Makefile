
HUGO=hugo
PORT=1313
BASEURL=http://localhost:$(PORT)/ansonwang

dev:
	$(HUGO) server \
		--baseURL=$(BASEURL) \
		--appendPort=false \
		--disableFastRender
