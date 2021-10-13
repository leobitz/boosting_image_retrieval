@ECHO OFF
for %%v in (resnet50 resnet101) do ( REM pca resent18 resent50 resent101 vit
	for %%l in (0.001 0.0001) do (
		for %%t in (single double) do (
            for %%r in (1 2 3 4 5) do (
                if exist result\pos\%%v-%%l-%%t-%%r (
                    echo file exists
                ) else (
                    echo result\pos\%%v-%%l-%%t-%%r 
                    python .\train.py --model_type %%t --vec_type %%v --learning_rate %%l --run %%r 
                )
			)
		)
	)
)
