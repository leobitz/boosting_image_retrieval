@ECHO OFF
for %%v in (resnet18 resnet50 resnet101 vit pca) do ( @REM 
	for %%l in (0.001 0.0001) do (
		for %%t in (single double) do (
            for %%r in (1 2 3 4 5) do (
                if exist result\trains\%%t-%%v-%%l-%%r (
                    echo file exists
                ) else (
                    if exist result\saves\%%t-%%v-%%l-%%r (
                        echo folder exists
                        @RD /S /Q "result\saves\%%t-%%v-%%l-%%r"
                    ) 
                    echo result\pos\%%t-%%v-%%l-%%r
                    python .\train.py --model_type %%t --vec_type %%v --learning_rate %%l --run %%r 
                    copy NUL result\trains\%%t-%%v-%%l-%%r
                )
			)
		)
	)
)
