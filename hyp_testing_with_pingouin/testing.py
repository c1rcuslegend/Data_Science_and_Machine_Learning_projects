import pandas as pd
import pingouin

men = pd.read_csv("./men_results.csv", parse_dates=["date"])
women = pd.read_csv("./women_results.csv", parse_dates=["date"])

men = men[(men["date"]>"2002-01-01") & (men["tournament"]=="FIFA World Cup")]
women = women[(women["date"]>"2002-01-01") & (women["tournament"]=="FIFA World Cup")]

men["group"] = "men"
women["group"] = "women"

men["total_score"] = men["home_score"]+men["away_score"]
women["total_score"] = women["home_score"]+women["away_score"]

glob = pd.concat([men, women], axis=0, ignore_index=True)[["total_score","group"]]
glob = glob.pivot(columns="group", values="total_score")

result = pingouin.mwu(x=glob["women"],
                      y=glob["men"],
                      alternative="greater")

p_val = result["p-val"].item()
result = "reject" if p_val<=0.01 else "fail to reject"

result_dict = {"p_val": p_val, "result": result}