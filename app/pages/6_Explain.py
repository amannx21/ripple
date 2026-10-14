import streamlit as st

from common import badge, current_instance, data_badges, instance_tag, money, page_setup, plan_for, safe, sidebar

page_setup("Explain")

ICON = {"SHIP": "🚚", "DISCOUNT": "🏷️", "HOLD": "🏬", "DIVERT": "🏭"}


@st.cache_data(show_spinner="Re-solving one counterfactual per batch…")
def cards_for(tag: str, _inst, _risk, _plan):
    from freshroute.explain import counterfactual_cards

    return counterfactual_cards(_inst, _risk, _plan)


def body():
    from freshroute.explain import discount_timing, primary_action

    ctl = sidebar()
    st.title("6 · Decision cards")
    inst, _ = current_instance(ctl)
    st.markdown(data_badges(inst), unsafe_allow_html=True)
    risk, plan, viol = plan_for(inst, ctl["alpha"], instance_tag(ctl))
    st.markdown("Every card shows the recommendation, the reason in plain language, and a **counterfactual**: the "
                "planner is re-solved with this batch barred from its main action, and the change in expected net "
                "revenue and Spoilage-at-Risk is reported.")
    cards = cards_for(instance_tag(ctl) + f"|{ctl['alpha']:.3f}", inst, risk, plan)
    actions = sorted({primary_action(plan, c["batch_id"])[0] for c in cards})
    pick = st.multiselect("Filter by main action", actions, default=actions)
    for c in cards:
        act = primary_action(plan, c["batch_id"])[0]
        if act not in pick:
            continue
        with st.container(border=True):
            top = st.columns([3, 1])
            top[0].markdown(f"#### {ICON.get(act, '')} {c['batch_id']} · {c['commodity']}: {c['recommendation']}")
            d = c["delta_expected_net_inr"]
            if d == d and d is not None:
                top[1].metric("Next-best alternative", money(d), "expected net vs recommended", delta_color="normal")
            else:
                top[1].markdown(badge("No feasible alternative", "fallback"), unsafe_allow_html=True)
            for r in c["reasons"]:
                st.markdown(f"- {r}")
            sar_now = "–" if c["sar_shipped_now"] is None else f"{c['sar_shipped_now']:.1%}"
            sar_alt = "–" if c["sar_shipped_alternative"] is None else f"{c['sar_shipped_alternative']:.1%}"
            st.markdown(f"**Counterfactual ({c['alternative']}):** {c['alternative_plan']}  \n"
                        f"Shipped-kg-weighted SaR: recommended {sar_now} → alternative {sar_alt}.")
            ship_rows = plan.rows[(plan.rows.batch_id == c["batch_id"]) & plan.rows.action.isin(["SHIP", "DISCOUNT"])]
            if not ship_rows.empty and (c["batch_id"], ship_rows.iloc[0].dest_id) in risk.samples:
                with st.expander("Discount timing: markdown now or wait one cycle?"):
                    dt = discount_timing(inst, risk, c["batch_id"], ship_rows.iloc[0].dest_id)
                    st.dataframe(dt.round(2), hide_index=True, use_container_width=True)
                    best_now = dt["value/kg now"].max()
                    best_later = dt["value/kg after 1-day hold"].max()
                    k = dt.loc[dt["value/kg now"].idxmax(), "discount"]
                    verdict = (f"Act now ({k:.0%} markdown) is worth ₹{best_now - best_later:,.1f}/kg more than waiting."
                               if best_now >= best_later else
                               f"Waiting one cycle is worth ₹{best_later - best_now:,.1f}/kg more (capacity permitting).")
                    st.markdown(verdict)
                    st.caption("Waiting shifts the Monte Carlo arrival-life samples down by the life lost during one "
                               "day at hub storage temperature, and adds the holding cost.")


safe(body)
