@ECHO OFF
for %%v in (vit) do ( @REM pca resent18 resent50 resent101 vit
	for %%l in (0.001 ) do (
		for %%t in (single double) do (
            for %%r in (1 2 3 4 5) do (
                @REM if exist result\trains\%%t-%%v-%%l-%%r (
                @REM     echo file exists
                @REM ) else (
                    
                    echo result\pos\%%t-%%v-%%l-%%r
                    python .\evaluate.py --model_type %%t --vec_type %%v --learning_rate %%l --run %%r 
                    @REM copy NUL result\trains\%%t-%%v-%%l-%%r
                @REM )
			)
		)
	)
)
