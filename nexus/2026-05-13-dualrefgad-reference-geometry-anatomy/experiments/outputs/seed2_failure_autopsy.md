# Seed2 Failure Autopsy — margin vs mat_mean

Runtime: 222.5s

## Case summaries

| case | n | anom ratio | mean margin | mean mat_mean | mean R_a purity | mean 1-hop anomaly ratio | mean mat std |
|---|---:|---:|---:|---:|---:|---:|---:|
| margin_high_mat_low | 25 | 1.000 | 0.9944 | 0.4868 | 0.880 | 0.000 | 0.6057 |
| mat_mean_false_positive | 25 | 0.000 | 0.9980 | 0.9997 | 0.150 | 0.040 | 0.0003 |
| anomaly_margin_wins_mat_loses | 25 | 1.000 | 1.0000 | 0.9942 | 1.000 | 0.000 | 0.0324 |

## Selected nodes

### margin_high_mat_low — node 20785 (label=1)
- scores: margin=0.9885 rank=3616; mat_mean=0.3502 rank=37929; rank_gap=34313
- R_a purity diagnostic: 0.250; R_n normal ratio: 1.000
- response shape: ra_std=0.2615, ra_q90=0.9824, mat_std=0.7671, mat_q90=0.9940, mat_high08=0.547
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 17818(y=0,v=0.901), 31069(y=0,v=0.853), 20798(y=0,v=0.749), 32852(y=0,v=0.686), 39860(y=0,v=0.517)

### margin_high_mat_low — node 24806 (label=1)
- scores: margin=0.9867 rank=3902; mat_mean=0.3572 rank=37672; rank_gap=33770
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.1249, ra_q90=0.9588, mat_std=0.7754, mat_q90=0.9931, mat_high08=0.578
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24551(y=1,v=0.985), 31811(y=1,v=0.985), 24661(y=1,v=0.985), 32443(y=1,v=0.983), 24637(y=1,v=0.980)

### margin_high_mat_low — node 10737 (label=1)
- scores: margin=0.9976 rank=1885; mat_mean=0.4194 rank=34934; rank_gap=33049
- R_a purity diagnostic: 0.500; R_n normal ratio: 1.000
- response shape: ra_std=0.0101, ra_q90=0.9992, mat_std=0.6394, mat_q90=0.9999, mat_high08=0.453
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 15489(y=0,v=0.846), 24111(y=1,v=0.608), 32302(y=0,v=0.546), 32668(y=0,v=0.546), 32072(y=0,v=0.537)

### margin_high_mat_low — node 39590 (label=1)
- scores: margin=0.9992 rank=1369; mat_mean=0.4712 rank=31907; rank_gap=30538
- R_a purity diagnostic: 0.500; R_n normal ratio: 1.000
- response shape: ra_std=0.0004, ra_q90=0.9989, mat_std=0.7183, mat_q90=0.9999, mat_high08=0.547
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 32483(y=0,v=0.653), 22916(y=1,v=0.652), 34160(y=1,v=0.561), 32668(y=0,v=0.534), 32302(y=0,v=0.534)

### margin_high_mat_low — node 24534 (label=1)
- scores: margin=0.9928 rank=2829; mat_mean=0.4786 rank=31382; rank_gap=28553
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0418, ra_q90=0.9802, mat_std=0.5783, mat_q90=0.9988, mat_high08=0.547
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24836(y=1,v=0.968), 24806(y=1,v=0.494), 20486(y=1,v=0.442), 24784(y=1,v=0.442), 20522(y=1,v=0.442)

### margin_high_mat_low — node 18537 (label=1)
- scores: margin=0.9994 rank=1325; mat_mean=0.4871 rank=30781; rank_gap=29456
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0001, ra_q90=0.9993, mat_std=0.5430, mat_q90=1.0000, mat_high08=0.531
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24836(y=1,v=0.975), 20486(y=1,v=0.455), 24784(y=1,v=0.455), 20542(y=1,v=0.455), 24681(y=1,v=0.455)

### margin_high_mat_low — node 18556 (label=1)
- scores: margin=0.9999 rank=703; mat_mean=0.4899 rank=30600; rank_gap=29897
- R_a purity diagnostic: 0.688; R_n normal ratio: 1.000
- response shape: ra_std=0.0224, ra_q90=0.9999, mat_std=0.7185, mat_q90=0.9987, mat_high08=0.531
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24628(y=0,v=0.746), 19854(y=1,v=0.745), 13001(y=0,v=0.518), 12317(y=0,v=0.511), 12431(y=0,v=0.503)

### margin_high_mat_low — node 17762 (label=1)
- scores: margin=0.9999 rank=646; mat_mean=0.4902 rank=30582; rank_gap=29936
- R_a purity diagnostic: 0.688; R_n normal ratio: 1.000
- response shape: ra_std=0.0223, ra_q90=0.9999, mat_std=0.7195, mat_q90=0.9987, mat_high08=0.531
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 19854(y=1,v=0.752), 24628(y=0,v=0.746), 13001(y=0,v=0.522), 12317(y=0,v=0.514), 12431(y=0,v=0.507)

### margin_high_mat_low — node 28718 (label=1)
- scores: margin=0.9999 rank=704; mat_mean=0.4902 rank=30578; rank_gap=29874
- R_a purity diagnostic: 0.688; R_n normal ratio: 1.000
- response shape: ra_std=0.0224, ra_q90=0.9999, mat_std=0.7182, mat_q90=0.9987, mat_high08=0.531
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 19854(y=1,v=0.749), 24628(y=0,v=0.746), 13001(y=0,v=0.519), 12317(y=0,v=0.511), 12431(y=0,v=0.504)

### margin_high_mat_low — node 33053 (label=1)
- scores: margin=0.9999 rank=679; mat_mean=0.4904 rank=30567; rank_gap=29888
- R_a purity diagnostic: 0.688; R_n normal ratio: 1.000
- response shape: ra_std=0.0222, ra_q90=0.9998, mat_std=0.7198, mat_q90=0.9987, mat_high08=0.531
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 19854(y=1,v=0.752), 24628(y=0,v=0.746), 13001(y=0,v=0.525), 12317(y=0,v=0.517), 12431(y=0,v=0.509)

### margin_high_mat_low — node 16098 (label=1)
- scores: margin=0.9947 rank=2480; mat_mean=0.4980 rank=30026; rank_gap=27546
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0008, ra_q90=0.9940, mat_std=0.5657, mat_q90=0.9996, mat_high08=0.562
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24534(y=1,v=0.999), 24836(y=1,v=0.972), 20486(y=1,v=0.428), 24784(y=1,v=0.428), 24681(y=1,v=0.428)

### margin_high_mat_low — node 21266 (label=1)
- scores: margin=0.9926 rank=2854; mat_mean=0.5003 rank=29861; rank_gap=27007
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0017, ra_q90=0.9918, mat_std=0.5631, mat_q90=0.9994, mat_high08=0.562
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24534(y=1,v=0.998), 24836(y=1,v=0.973), 20486(y=1,v=0.431), 24784(y=1,v=0.431), 24681(y=1,v=0.431)

### margin_high_mat_low — node 16454 (label=1)
- scores: margin=0.9945 rank=2527; mat_mean=0.5041 rank=29576; rank_gap=27049
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0009, ra_q90=0.9938, mat_std=0.5586, mat_q90=0.9996, mat_high08=0.562
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24534(y=1,v=0.999), 24836(y=1,v=0.969), 20486(y=1,v=0.436), 24784(y=1,v=0.436), 20522(y=1,v=0.436)

### margin_high_mat_low — node 24911 (label=1)
- scores: margin=0.9994 rank=1303; mat_mean=0.5099 rank=29152; rank_gap=27849
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0001, ra_q90=0.9994, mat_std=0.5192, mat_q90=1.0000, mat_high08=0.531
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24836(y=1,v=0.979), 20486(y=1,v=0.479), 24784(y=1,v=0.479), 20542(y=1,v=0.479), 24681(y=1,v=0.479)

### margin_high_mat_low — node 20916 (label=1)
- scores: margin=0.9927 rank=2848; mat_mean=0.5079 rank=29308; rank_gap=26460
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0017, ra_q90=0.9918, mat_std=0.5550, mat_q90=0.9994, mat_high08=0.562
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24534(y=1,v=0.998), 24836(y=1,v=0.974), 20486(y=1,v=0.440), 24784(y=1,v=0.440), 24681(y=1,v=0.440)

### margin_high_mat_low — node 18239 (label=1)
- scores: margin=0.9996 rank=1260; mat_mean=0.5160 rank=28688; rank_gap=27428
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0001, ra_q90=0.9996, mat_std=0.5131, mat_q90=1.0000, mat_high08=0.531
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24836(y=1,v=0.989), 20486(y=1,v=0.485), 24784(y=1,v=0.485), 24681(y=1,v=0.485), 20542(y=1,v=0.485)

### margin_high_mat_low — node 18741 (label=1)
- scores: margin=0.9996 rank=1259; mat_mean=0.5164 rank=28655; rank_gap=27396
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0001, ra_q90=0.9996, mat_std=0.5130, mat_q90=1.0000, mat_high08=0.531
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24836(y=1,v=0.992), 20486(y=1,v=0.485), 24784(y=1,v=0.485), 24681(y=1,v=0.485), 20542(y=1,v=0.485)

### margin_high_mat_low — node 18519 (label=1)
- scores: margin=0.9996 rank=1258; mat_mean=0.5172 rank=28599; rank_gap=27341
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0001, ra_q90=0.9996, mat_std=0.5119, mat_q90=1.0000, mat_high08=0.531
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24836(y=1,v=0.989), 20486(y=1,v=0.486), 24784(y=1,v=0.486), 24681(y=1,v=0.486), 20542(y=1,v=0.486)

### margin_high_mat_low — node 24587 (label=1)
- scores: margin=0.9833 rank=4422; mat_mean=0.5013 rank=29787; rank_gap=25365
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0292, ra_q90=0.9643, mat_std=0.5810, mat_q90=0.9976, mat_high08=0.578
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24534(y=1,v=0.998), 24836(y=1,v=0.963), 24806(y=1,v=0.500), 20486(y=1,v=0.428), 24784(y=1,v=0.428)

### margin_high_mat_low — node 23908 (label=1)
- scores: margin=0.9995 rank=1281; mat_mean=0.5176 rank=28568; rank_gap=27287
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0001, ra_q90=0.9995, mat_std=0.5116, mat_q90=1.0000, mat_high08=0.531
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24836(y=1,v=0.987), 20486(y=1,v=0.486), 24784(y=1,v=0.486), 24681(y=1,v=0.486), 20542(y=1,v=0.486)

### margin_high_mat_low — node 27516 (label=1)
- scores: margin=0.9950 rank=2430; mat_mean=0.5136 rank=28871; rank_gap=26441
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0007, ra_q90=0.9943, mat_std=0.5489, mat_q90=0.9996, mat_high08=0.562
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24534(y=1,v=0.997), 24836(y=1,v=0.980), 20486(y=1,v=0.446), 24784(y=1,v=0.446), 24681(y=1,v=0.446)

### margin_high_mat_low — node 28647 (label=1)
- scores: margin=0.9860 rank=4038; mat_mean=0.5072 rank=29362; rank_gap=25324
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0318, ra_q90=0.9682, mat_std=0.5794, mat_q90=0.9978, mat_high08=0.578
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24534(y=1,v=0.999), 24836(y=1,v=0.969), 24806(y=1,v=0.498), 20486(y=1,v=0.435), 24784(y=1,v=0.435)

### margin_high_mat_low — node 28317 (label=1)
- scores: margin=0.9859 rank=4041; mat_mean=0.5091 rank=29219; rank_gap=25178
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0317, ra_q90=0.9681, mat_std=0.5776, mat_q90=0.9979, mat_high08=0.578
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24534(y=1,v=0.999), 24836(y=1,v=0.969), 24806(y=1,v=0.497), 20486(y=1,v=0.437), 24784(y=1,v=0.437)

### margin_high_mat_low — node 33997 (label=1)
- scores: margin=0.9867 rank=3905; mat_mean=0.5136 rank=28873; rank_gap=24968
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0325, ra_q90=0.9692, mat_std=0.5726, mat_q90=0.9979, mat_high08=0.578
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24534(y=1,v=0.998), 24836(y=1,v=0.972), 24806(y=1,v=0.497), 20486(y=1,v=0.442), 24784(y=1,v=0.442)

### margin_high_mat_low — node 28844 (label=1)
- scores: margin=0.9862 rank=3998; mat_mean=0.5134 rank=28892; rank_gap=24894
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0319, ra_q90=0.9684, mat_std=0.5735, mat_q90=0.9979, mat_high08=0.578
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24534(y=1,v=0.999), 24836(y=1,v=0.971), 24806(y=1,v=0.497), 20486(y=1,v=0.442), 24784(y=1,v=0.442)

### mat_mean_false_positive — node 15447 (label=0)
- scores: margin=1.0000 rank=2; mat_mean=1.0000 rank=1; rank_gap=-1
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12564(y=0,v=1.000), 15227(y=0,v=1.000), 27499(y=0,v=1.000), 27462(y=0,v=1.000), 27322(y=0,v=1.000)

### mat_mean_false_positive — node 27098 (label=0)
- scores: margin=1.0000 rank=8; mat_mean=1.0000 rank=3; rank_gap=-5
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12564(y=0,v=1.000), 15227(y=0,v=1.000), 27566(y=0,v=1.000), 27042(y=0,v=1.000), 26908(y=0,v=1.000)

### mat_mean_false_positive — node 27520 (label=0)
- scores: margin=1.0000 rank=35; mat_mean=1.0000 rank=5; rank_gap=-30
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12564(y=0,v=1.000), 27499(y=0,v=1.000), 27462(y=0,v=1.000), 27322(y=0,v=1.000), 27303(y=0,v=1.000)

### mat_mean_false_positive — node 12564 (label=0)
- scores: margin=1.0000 rank=55; mat_mean=1.0000 rank=9; rank_gap=-46
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 15227(y=0,v=1.000), 15259(y=0,v=1.000), 27499(y=0,v=1.000), 27462(y=0,v=1.000), 27322(y=0,v=1.000)

### mat_mean_false_positive — node 27585 (label=0)
- scores: margin=1.0000 rank=39; mat_mean=1.0000 rank=25; rank_gap=-14
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12564(y=0,v=1.000), 27499(y=0,v=1.000), 27462(y=0,v=1.000), 27322(y=0,v=1.000), 27303(y=0,v=1.000)

### mat_mean_false_positive — node 27169 (label=0)
- scores: margin=1.0000 rank=22; mat_mean=1.0000 rank=33; rank_gap=11
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12564(y=0,v=1.000), 15227(y=0,v=1.000), 27566(y=0,v=1.000), 27042(y=0,v=1.000), 26908(y=0,v=1.000)

### mat_mean_false_positive — node 15227 (label=0)
- scores: margin=1.0000 rank=62; mat_mean=1.0000 rank=35; rank_gap=-27
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12564(y=0,v=1.000), 15259(y=0,v=1.000), 27499(y=0,v=1.000), 27462(y=0,v=1.000), 27322(y=0,v=1.000)

### mat_mean_false_positive — node 27303 (label=0)
- scores: margin=1.0000 rank=47; mat_mean=1.0000 rank=39; rank_gap=-8
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12564(y=0,v=1.000), 27499(y=0,v=1.000), 27462(y=0,v=1.000), 27322(y=0,v=1.000), 27585(y=0,v=1.000)

### mat_mean_false_positive — node 15259 (label=0)
- scores: margin=1.0000 rank=64; mat_mean=1.0000 rank=48; rank_gap=-16
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12564(y=0,v=1.000), 15227(y=0,v=1.000), 27566(y=0,v=1.000), 27042(y=0,v=1.000), 26908(y=0,v=1.000)

### mat_mean_false_positive — node 27462 (label=0)
- scores: margin=1.0000 rank=45; mat_mean=1.0000 rank=52; rank_gap=7
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12564(y=0,v=1.000), 27499(y=0,v=1.000), 27322(y=0,v=1.000), 27303(y=0,v=1.000), 27585(y=0,v=1.000)

### mat_mean_false_positive — node 27225 (label=0)
- scores: margin=1.0000 rank=16; mat_mean=1.0000 rank=53; rank_gap=37
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12564(y=0,v=1.000), 15227(y=0,v=1.000), 27566(y=0,v=1.000), 27042(y=0,v=1.000), 26908(y=0,v=1.000)

### mat_mean_false_positive — node 27499 (label=0)
- scores: margin=1.0000 rank=23; mat_mean=1.0000 rank=57; rank_gap=34
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12564(y=0,v=1.000), 27462(y=0,v=1.000), 27322(y=0,v=1.000), 27303(y=0,v=1.000), 27585(y=0,v=1.000)

### mat_mean_false_positive — node 27243 (label=0)
- scores: margin=1.0000 rank=50; mat_mean=1.0000 rank=74; rank_gap=24
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12564(y=0,v=1.000), 15227(y=0,v=1.000), 27566(y=0,v=1.000), 27042(y=0,v=1.000), 26908(y=0,v=1.000)

### mat_mean_false_positive — node 27322 (label=0)
- scores: margin=1.0000 rank=48; mat_mean=1.0000 rank=79; rank_gap=31
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12564(y=0,v=1.000), 27499(y=0,v=1.000), 27462(y=0,v=1.000), 27303(y=0,v=1.000), 27585(y=0,v=1.000)

### mat_mean_false_positive — node 26908 (label=0)
- scores: margin=1.0000 rank=73; mat_mean=1.0000 rank=27; rank_gap=-46
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12564(y=0,v=1.000), 27566(y=0,v=1.000), 27042(y=0,v=1.000), 15259(y=0,v=1.000), 27225(y=0,v=1.000)

### mat_mean_false_positive — node 27566 (label=0)
- scores: margin=1.0000 rank=70; mat_mean=1.0000 rank=43; rank_gap=-27
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12564(y=0,v=1.000), 27042(y=0,v=1.000), 26908(y=0,v=1.000), 15259(y=0,v=1.000), 27225(y=0,v=1.000)

### mat_mean_false_positive — node 27042 (label=0)
- scores: margin=1.0000 rank=67; mat_mean=1.0000 rank=73; rank_gap=6
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12564(y=0,v=1.000), 27566(y=0,v=1.000), 26908(y=0,v=1.000), 15259(y=0,v=1.000), 27225(y=0,v=1.000)

### mat_mean_false_positive — node 17393 (label=0)
- scores: margin=0.9993 rank=1354; mat_mean=0.9997 rank=249; rank_gap=-1105
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0018, ra_q90=0.9992, mat_std=0.0005, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=1.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 26458(y=1,v=1.000), 26202(y=1,v=1.000), 25813(y=1,v=1.000), 25786(y=1,v=1.000), 27252(y=1,v=1.000)

### mat_mean_false_positive — node 20641 (label=0)
- scores: margin=0.9968 rank=2082; mat_mean=0.9995 rank=505; rank_gap=-1577
- R_a purity diagnostic: 0.938; R_n normal ratio: 1.000
- response shape: ra_std=0.1143, ra_q90=0.9996, mat_std=0.0016, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 19766(y=1,v=1.000), 24865(y=1,v=1.000), 24886(y=1,v=1.000), 24807(y=1,v=1.000), 20488(y=1,v=1.000)

### mat_mean_false_positive — node 21567 (label=0)
- scores: margin=0.9978 rank=1845; mat_mean=0.9995 rank=506; rank_gap=-1339
- R_a purity diagnostic: 0.938; R_n normal ratio: 1.000
- response shape: ra_std=0.1144, ra_q90=0.9997, mat_std=0.0017, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 20433(y=1,v=1.000), 24807(y=1,v=1.000), 24558(y=1,v=1.000), 16140(y=1,v=1.000), 25168(y=1,v=1.000)

### mat_mean_false_positive — node 17567 (label=0)
- scores: margin=0.9990 rank=1413; mat_mean=0.9988 rank=676; rank_gap=-737
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0005, ra_q90=0.9996, mat_std=0.0005, mat_q90=0.9996, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12709(y=0,v=1.000), 12434(y=0,v=1.000), 13100(y=0,v=1.000), 13141(y=0,v=0.999), 12862(y=0,v=0.998)

### mat_mean_false_positive — node 12709 (label=0)
- scores: margin=0.9990 rank=1414; mat_mean=0.9988 rank=677; rank_gap=-737
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0005, ra_q90=0.9996, mat_std=0.0005, mat_q90=0.9996, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 17567(y=0,v=1.000), 12434(y=0,v=1.000), 13100(y=0,v=1.000), 13141(y=0,v=0.999), 24009(y=0,v=0.998)

### mat_mean_false_positive — node 10424 (label=0)
- scores: margin=0.9989 rank=1437; mat_mean=0.9987 rank=681; rank_gap=-756
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0005, ra_q90=0.9994, mat_std=0.0005, mat_q90=0.9994, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 12434(y=0,v=1.000), 12709(y=0,v=0.999), 17567(y=0,v=0.999), 13100(y=0,v=0.999), 13141(y=0,v=0.999)

### mat_mean_false_positive — node 12434 (label=0)
- scores: margin=0.9988 rank=1471; mat_mean=0.9985 rank=698; rank_gap=-773
- R_a purity diagnostic: 0.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0006, ra_q90=0.9996, mat_std=0.0006, mat_q90=0.9997, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 10424(y=0,v=1.000), 12709(y=0,v=1.000), 17567(y=0,v=1.000), 13100(y=0,v=0.999), 13141(y=0,v=0.999)

### mat_mean_false_positive — node 23295 (label=0)
- scores: margin=0.9609 rank=7350; mat_mean=0.9984 rank=701; rank_gap=-6649
- R_a purity diagnostic: 0.875; R_n normal ratio: 1.000
- response shape: ra_std=0.1304, ra_q90=0.9667, mat_std=0.0005, mat_q90=0.9987, mat_high08=1.000
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 31659(y=1,v=0.999), 24865(y=1,v=0.999), 24558(y=1,v=0.999), 16140(y=1,v=0.999), 24807(y=1,v=0.999)

### anomaly_margin_wins_mat_loses — node 11203 (label=1)
- scores: margin=1.0000 rank=1; mat_mean=1.0000 rank=58; rank_gap=57
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.939
- top anomaly refs by matrix column mean: 11483(y=1,v=1.000), 11123(y=1,v=1.000), 11069(y=1,v=1.000), 11366(y=1,v=1.000), 11399(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 11462 (label=1)
- scores: margin=1.0000 rank=3; mat_mean=1.0000 rank=63; rank_gap=60
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=1.000
- top anomaly refs by matrix column mean: 11200(y=1,v=1.000), 11271(y=1,v=1.000), 11312(y=1,v=1.000), 10940(y=1,v=1.000), 11254(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 14035 (label=1)
- scores: margin=1.0000 rank=4; mat_mean=1.0000 rank=45; rank_gap=41
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.739
- top anomaly refs by matrix column mean: 14139(y=1,v=1.000), 13809(y=1,v=1.000), 13895(y=1,v=1.000), 13743(y=1,v=1.000), 14010(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 11196 (label=1)
- scores: margin=1.0000 rank=5; mat_mean=1.0000 rank=62; rank_gap=57
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=1.000
- top anomaly refs by matrix column mean: 11200(y=1,v=1.000), 11271(y=1,v=1.000), 11312(y=1,v=1.000), 11254(y=1,v=1.000), 10816(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 11439 (label=1)
- scores: margin=1.0000 rank=7; mat_mean=1.0000 rank=38; rank_gap=31
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=1.000
- top anomaly refs by matrix column mean: 11271(y=1,v=1.000), 11312(y=1,v=1.000), 10940(y=1,v=1.000), 11254(y=1,v=1.000), 10816(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 11312 (label=1)
- scores: margin=1.0000 rank=9; mat_mean=1.0000 rank=60; rank_gap=51
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=1.000
- top anomaly refs by matrix column mean: 11200(y=1,v=1.000), 11271(y=1,v=1.000), 10940(y=1,v=1.000), 11462(y=1,v=1.000), 11254(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 13806 (label=1)
- scores: margin=1.0000 rank=10; mat_mean=1.0000 rank=40; rank_gap=30
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.739
- top anomaly refs by matrix column mean: 14139(y=1,v=1.000), 13895(y=1,v=1.000), 13794(y=1,v=1.000), 13743(y=1,v=1.000), 14299(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 11406 (label=1)
- scores: margin=1.0000 rank=12; mat_mean=1.0000 rank=31; rank_gap=19
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=1.000
- top anomaly refs by matrix column mean: 11271(y=1,v=1.000), 11312(y=1,v=1.000), 10940(y=1,v=1.000), 11254(y=1,v=1.000), 10816(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 13809 (label=1)
- scores: margin=1.0000 rank=13; mat_mean=1.0000 rank=29; rank_gap=16
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.739
- top anomaly refs by matrix column mean: 14139(y=1,v=1.000), 13895(y=1,v=1.000), 13743(y=1,v=1.000), 13794(y=1,v=1.000), 14010(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 10925 (label=1)
- scores: margin=1.0000 rank=14; mat_mean=1.0000 rank=69; rank_gap=55
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.939
- top anomaly refs by matrix column mean: 11366(y=1,v=1.000), 11483(y=1,v=1.000), 11123(y=1,v=1.000), 11069(y=1,v=1.000), 10879(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 10762 (label=1)
- scores: margin=1.0000 rank=19; mat_mean=1.0000 rank=67; rank_gap=48
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.939
- top anomaly refs by matrix column mean: 11366(y=1,v=1.000), 10879(y=1,v=1.000), 11483(y=1,v=1.000), 11123(y=1,v=1.000), 11069(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 11069 (label=1)
- scores: margin=1.0000 rank=20; mat_mean=1.0000 rank=66; rank_gap=46
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.939
- top anomaly refs by matrix column mean: 11123(y=1,v=1.000), 11366(y=1,v=1.000), 10879(y=1,v=1.000), 11399(y=1,v=1.000), 10746(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 13891 (label=1)
- scores: margin=1.0000 rank=21; mat_mean=1.0000 rank=98; rank_gap=77
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.885
- top anomaly refs by matrix column mean: 14431(y=1,v=1.000), 13973(y=1,v=1.000), 14315(y=1,v=1.000), 13925(y=1,v=1.000), 14373(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 11130 (label=1)
- scores: margin=1.0000 rank=26; mat_mean=1.0000 rank=70; rank_gap=44
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=1.000
- top anomaly refs by matrix column mean: 11271(y=1,v=1.000), 11312(y=1,v=1.000), 10940(y=1,v=1.000), 11254(y=1,v=1.000), 10816(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 13925 (label=1)
- scores: margin=1.0000 rank=29; mat_mean=1.0000 rank=101; rank_gap=72
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.885
- top anomaly refs by matrix column mean: 13973(y=1,v=1.000), 14431(y=1,v=1.000), 14373(y=1,v=1.000), 14315(y=1,v=1.000), 13891(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 11309 (label=1)
- scores: margin=1.0000 rank=31; mat_mean=1.0000 rank=80; rank_gap=49
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=1.000
- top anomaly refs by matrix column mean: 11200(y=1,v=1.000), 11271(y=1,v=1.000), 11254(y=1,v=1.000), 10993(y=1,v=1.000), 11439(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 11382 (label=1)
- scores: margin=1.0000 rank=32; mat_mean=1.0000 rank=72; rank_gap=40
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.939
- top anomaly refs by matrix column mean: 11483(y=1,v=1.000), 11123(y=1,v=1.000), 11069(y=1,v=1.000), 11399(y=1,v=1.000), 11366(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 14139 (label=1)
- scores: margin=1.0000 rank=38; mat_mean=1.0000 rank=77; rank_gap=39
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.739
- top anomaly refs by matrix column mean: 13809(y=1,v=1.000), 13895(y=1,v=1.000), 13794(y=1,v=1.000), 13743(y=1,v=1.000), 14010(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 14431 (label=1)
- scores: margin=1.0000 rank=40; mat_mean=1.0000 rank=100; rank_gap=60
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.885
- top anomaly refs by matrix column mean: 13891(y=1,v=1.000), 13973(y=1,v=1.000), 14373(y=1,v=1.000), 14315(y=1,v=1.000), 13925(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 14315 (label=1)
- scores: margin=1.0000 rank=49; mat_mean=1.0000 rank=99; rank_gap=50
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0000, ra_q90=1.0000, mat_std=0.0000, mat_q90=1.0000, mat_high08=1.000
- local graph: degree=1, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.885
- top anomaly refs by matrix column mean: 13973(y=1,v=1.000), 14431(y=1,v=1.000), 14373(y=1,v=1.000), 13925(y=1,v=1.000), 13891(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 20522 (label=1)
- scores: margin=1.0000 rank=103; mat_mean=0.9709 rank=1458; rank_gap=1355
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0001, ra_q90=1.0000, mat_std=0.1618, mat_q90=1.0000, mat_high08=0.969
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 20319(y=1,v=1.000), 24681(y=1,v=1.000), 20337(y=1,v=1.000), 24680(y=1,v=1.000), 24823(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 20324 (label=1)
- scores: margin=1.0000 rank=104; mat_mean=0.9709 rank=1465; rank_gap=1361
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0001, ra_q90=1.0000, mat_std=0.1618, mat_q90=1.0000, mat_high08=0.969
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 20319(y=1,v=1.000), 20337(y=1,v=1.000), 24823(y=1,v=1.000), 24813(y=1,v=1.000), 24784(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 24680 (label=1)
- scores: margin=1.0000 rank=105; mat_mean=0.9709 rank=1464; rank_gap=1359
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0001, ra_q90=1.0000, mat_std=0.1618, mat_q90=1.0000, mat_high08=0.969
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 20319(y=1,v=1.000), 24673(y=1,v=1.000), 24823(y=1,v=1.000), 24784(y=1,v=1.000), 20482(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 20486 (label=1)
- scores: margin=1.0000 rank=106; mat_mean=0.9709 rank=1455; rank_gap=1349
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0001, ra_q90=1.0000, mat_std=0.1618, mat_q90=1.0000, mat_high08=0.969
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24680(y=1,v=1.000), 24813(y=1,v=1.000), 24510(y=1,v=1.000), 20542(y=1,v=1.000), 24673(y=1,v=1.000)

### anomaly_margin_wins_mat_loses — node 20542 (label=1)
- scores: margin=1.0000 rank=107; mat_mean=0.9709 rank=1457; rank_gap=1350
- R_a purity diagnostic: 1.000; R_n normal ratio: 1.000
- response shape: ra_std=0.0001, ra_q90=1.0000, mat_std=0.1618, mat_q90=1.0000, mat_high08=0.969
- local graph: degree=0, 1hop_anom_ratio=0.000, 2hop_anom_ratio=0.000
- top anomaly refs by matrix column mean: 24673(y=1,v=1.000), 24681(y=1,v=1.000), 24680(y=1,v=1.000), 20324(y=1,v=1.000), 24823(y=1,v=1.000)

