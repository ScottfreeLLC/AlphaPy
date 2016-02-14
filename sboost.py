##############################################################
#
# Package   : AlphaPy
# Module    : sboost
# Version   : 1.0
# Copyright : Mark Conway
# Date      : June 29, 2013
#
##############################################################


def sboost(group, system)
    function(group,
               system,
               fractal = "d1",
               outputs = "return",
               inputs = ".",
               lag = 1,
               groupby = c("month", "year"),
               ...)
{
    gspace = newspace("systems", "trades", fractal)
    group.space = gspace
    alltrades = gapply(group, mergeFrames, system)
    portfolio = generateportfolio(group, alltrades, 0, restricted = FALSE, ...)
    gspace.what = "position"
    gspace.wherein = "states"
    group.space = gspace   # now we move to position states
    splitframe(group)
    aspace = newspace(system, "sboost", fractal)
    method = new("mrmethod")
    for (i in seq(along = outputs))
    {
        response = responses[i]
        sb = predict(group,
                           output[i],
                           inputs,
                           method,
                           lag,
                           groupby,
                                   aspace)
        if (exists(sb))
            cat("System Boost", getname(sb), "is complete\n")
        else
            cat("System Boost for", response, "did not complete\n")
    }
    rm(gspace, aspace)
}