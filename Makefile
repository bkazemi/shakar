.PHONY: rebundle web-commit unified-hl-deploy highlight-deploy

rebundle:
	python web/bundle.py > web/shakar_bundle.py

rebundle-commit: rebundle
	git add web/shakar_bundle.py && git commit -m 'chore(web): rebundle' --only -- web/shakar_bundle.py

web-commit: rebundle
	git add -u web/
	git commit

unified-hl-deploy:
	make -C unified_hl all && make -C unified_hl deploy

highlight-deploy:
	make -C highlight all && make -C highlight deploy
