##############################################################
#
# Package   : AlphaPy
# Module    : pboost
# Version   : 1.0
# Copyright : Mark Conway
# Date      : June 29, 2013
#
##############################################################


pboost =
    function(group,
               systems,
               nruns = 1000,
               fractal = "d1",
               kovars = "-profit",
               wtvars = "volatility",
               response = "value",
               lag = 0,
               ...)
{
    gspace = newspace("systems", "trades", fractal)
    group.space = gspace
    gmembers = gselect(group, "member")
    glen = length(gmembers)
    alltrades = gapply(group, mergeframes, systems)
    sbperiod = sample(1:glen, nruns, replace = TRUE)
    smargin = runif(nruns, max = 2)
    smincash = runif(nruns)
    smaxpos = sample(1:gen, nruns, replace = TRUE)
    smaxloss = runif(nruns)
    skopos = sample(0:maxpos, nruns, replace = TRUE)
    skoby = sample(1:length(kovars), nruns, replace = TRUE)
    sweightby = sample(1:length(wtvars), runs, replace = TRUE)
    scoin = sample(0:1, nruns, replace = TRUE)
    pname = paste(nparse(group, "oid"), "portfolio", sep = ".")
    sspace = newspace("pboost", "states", fractal)
    pbstates = newstates(pname, sspace, "portfolio", sep = ".")
    pspace = newspace("group", "states", fractal)
    for (i in seq(along = nruns))
    {
        cminus = ifelse(scoin[i], "", "-")   # 1 is +, 0 is -
        portfolio =
            generateportfolio(
                group,
                alltrades,
                bperiod = sbperiod[i],
                margin = smargin[i],
                mincash = smincash[i],
                maxpos = smaxpos[i],
                maxloss = smaxloss[i],
                kopos = skopos[i],
                koby = paste(cminus, kovars[skoby[i]]),
                weightby = paste(cminus, wtvars[sweightby[i]]))
        pfstates = getframe(pname, pframe)
        fpv = pfstates[nrow(pfstates), ]
        pbstates[i, ] = fpv
        deleteportfolio(portfolio)
    }
    pb = analyze(pname,
                               sframe,
                               ...)
    if (exists(pb))
        cat("Portfolio Boost", pname, "is complete\n")
    else
        cat("Portfolio Boost for", group.name, "did not complete\n")
    rm(gspace, sspace, pspace)
}