import streamlit as st
import pandas as pd
from pathlib import Path


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)


# ==================================================
# PROJECT PATHS
# ==================================================

dashboard_folder = Path(__file__).resolve().parent
project_folder = dashboard_folder.parent
data_folder = project_folder / "data"

DATA_PATH = data_folder / "customer_churn_predictions.csv"


# ==================================================
# LOAD DATA
# ==================================================

dashboard_data = pd.read_csv(DATA_PATH)


# ==================================================
# MODEL / BUSINESS CONSTANTS
# ==================================================

SELECTED_THRESHOLD = 0.35

MODEL_ACCURACY = 0.778566
MODEL_PRECISION = 0.564854
MODEL_RECALL = 0.721925
MODEL_F1 = 0.633803
MODEL_ROC_AUC = 0.848097

CAMPAIGN_COST_PER_CUSTOMER = 500
RETAINED_CUSTOMER_VALUE = 5000


# ==================================================
# KPI CALCULATIONS
# ==================================================

total_customers = len(dashboard_data)

high_risk_customers = (
    dashboard_data["Risk Level"] == "High Risk"
).sum()

high_risk_percentage = (
    high_risk_customers / total_customers
) * 100

average_churn_probability = (
    dashboard_data["Churn Probability"].mean()
) * 100

observed_churn_rate = (
    dashboard_data["Actual Churn"].mean()
) * 100

actual_churners_flagged = (
    (
        (dashboard_data["Risk Level"] == "High Risk")
        & (dashboard_data["Actual Churn"] == 1)
    )
    .sum()
)


# ==================================================
# PREPARE CHART DATA
# ==================================================

risk_distribution = (
    dashboard_data["Risk Level"]
    .value_counts()
    .rename_axis("Risk Level")
    .reset_index(name="Customers")
)

risk_churn_rate = (
    dashboard_data
    .groupby("Risk Level")["Actual Churn"]
    .mean()
    .mul(100)
    .reset_index(name="Observed Churn Rate")
)

contract_churn_rate = (
    dashboard_data
    .groupby("Contract")["Actual Churn"]
    .mean()
    .mul(100)
    .reset_index(name="Observed Churn Rate")
)

internet_churn_rate = (
    dashboard_data
    .groupby("Internet Service")["Actual Churn"]
    .mean()
    .mul(100)
    .reset_index(name="Observed Churn Rate")
)

payment_churn_rate = (
    dashboard_data
    .groupby("Payment Method")["Actual Churn"]
    .mean()
    .mul(100)
    .reset_index(name="Observed Churn Rate")
)


# ==================================================
# PAGE HEADER
# ==================================================

st.title(
    "Customer Churn Prediction & Retention Analytics"
)

st.write(
    "Machine learning dashboard for customer churn "
    "prediction and retention analysis."
)


# ==================================================
# EVALUATION NOTE
# ==================================================

st.info(
    "This dashboard uses the held-out test dataset. "
    "Actual Churn is available here for model evaluation only; "
    "it would not be available when making a real-time prediction."
)


# ==================================================
# MAIN TABS
# ==================================================

overview_tab, risk_tab, retention_tab, impact_tab = st.tabs(
    [
        "Overview",
        "Risk Explorer",
        "Retention Strategy",
        "Business Impact"
    ]
)


# ==================================================
# TAB 1 — OVERVIEW
# ==================================================

with overview_tab:

    st.subheader("Key Performance Indicators")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Customers",
            value=f"{total_customers:,}"
        )

    with col2:
        st.metric(
            label="High-Risk Customers",
            value=f"{high_risk_customers:,}"
        )

    with col3:
        st.metric(
            label="High-Risk %",
            value=f"{high_risk_percentage:.2f}%"
        )

    with col4:
        st.metric(
            label="Avg Churn Probability",
            value=f"{average_churn_probability:.2f}%"
        )

    with col5:
        st.metric(
            label="Actual Churners Flagged",
            value=f"{actual_churners_flagged:,}"
        )


    # ----------------------------------------------
    # Risk Analysis
    # ----------------------------------------------

    st.subheader("Risk Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Customer Risk Distribution**")

        st.bar_chart(
            risk_distribution,
            x="Risk Level",
            y="Customers"
        )

    with col2:

        st.markdown(
            "**Observed Churn Rate by Risk Level**"
        )

        st.bar_chart(
            risk_churn_rate,
            x="Risk Level",
            y="Observed Churn Rate"
        )


    # ----------------------------------------------
    # Customer Segmentation
    # ----------------------------------------------

    st.subheader("Customer Segmentation")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "**Observed Churn Rate by Contract**"
        )

        st.bar_chart(
            contract_churn_rate,
            x="Contract",
            y="Observed Churn Rate"
        )

    with col2:

        st.markdown(
            "**Observed Churn Rate by Internet Service**"
        )

        st.bar_chart(
            internet_churn_rate,
            x="Internet Service",
            y="Observed Churn Rate"
        )


    # ----------------------------------------------
    # Payment Analysis
    # ----------------------------------------------

    st.subheader("Payment Analysis")

    st.bar_chart(
        payment_churn_rate,
        x="Payment Method",
        y="Observed Churn Rate"
    )


    # ----------------------------------------------
    # Model Performance
    # ----------------------------------------------

    st.subheader("Final Model Performance")

    st.caption(
        "Final Logistic Regression model evaluated on the "
        "held-out test set using the selected 0.35 threshold."
    )

    model_col1, model_col2, model_col3, model_col4, model_col5 = (
        st.columns(5)
    )

    with model_col1:

        st.metric(
            label="Accuracy",
            value=f"{MODEL_ACCURACY:.2%}"
        )

    with model_col2:

        st.metric(
            label="Precision",
            value=f"{MODEL_PRECISION:.2%}"
        )

    with model_col3:

        st.metric(
            label="Recall",
            value=f"{MODEL_RECALL:.2%}"
        )

    with model_col4:

        st.metric(
            label="F1 Score",
            value=f"{MODEL_F1:.2%}"
        )

    with model_col5:

        st.metric(
            label="ROC-AUC",
            value=f"{MODEL_ROC_AUC:.2%}"
        )

    st.write(
        f"Selected probability threshold: "
        f"**{SELECTED_THRESHOLD:.2f}**"
    )


# ==================================================
# TAB 2 — RISK EXPLORER
# ==================================================

with risk_tab:

    st.subheader("Customer Risk Explorer")

    st.write(
        "Use the filters below to identify customer segments "
        "for retention analysis."
    )


    # ----------------------------------------------
    # Filter Controls
    # ----------------------------------------------

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:

        selected_risk = st.selectbox(
            "Risk Level",
            options=[
                "All"
            ] + sorted(
                dashboard_data["Risk Level"]
                .unique()
                .tolist()
            )
        )

    with filter_col2:

        selected_contract = st.selectbox(
            "Contract",
            options=[
                "All"
            ] + sorted(
                dashboard_data["Contract"]
                .unique()
                .tolist()
            )
        )

    with filter_col3:

        selected_internet = st.selectbox(
            "Internet Service",
            options=[
                "All"
            ] + sorted(
                dashboard_data["Internet Service"]
                .unique()
                .tolist()
            )
        )


    filter_col4, filter_col5 = st.columns(2)

    with filter_col4:

        selected_payment = st.selectbox(
            "Payment Method",
            options=[
                "All"
            ] + sorted(
                dashboard_data["Payment Method"]
                .unique()
                .tolist()
            )
        )

    with filter_col5:

        minimum_probability = st.slider(
            "Minimum Churn Probability",
            min_value=0.0,
            max_value=1.0,
            value=SELECTED_THRESHOLD,
            step=0.05
        )


    # ----------------------------------------------
    # Customer ID Search
    # ----------------------------------------------

    customer_search = st.text_input(
        "Search Customer ID",
        placeholder="Example: 5178-LMXOP"
    )


    # ----------------------------------------------
    # Apply Filters
    # ----------------------------------------------

    filtered_data = dashboard_data.copy()

    if selected_risk != "All":

        filtered_data = filtered_data[
            filtered_data["Risk Level"] == selected_risk
        ]

    if selected_contract != "All":

        filtered_data = filtered_data[
            filtered_data["Contract"] == selected_contract
        ]

    if selected_internet != "All":

        filtered_data = filtered_data[
            filtered_data["Internet Service"] == selected_internet
        ]

    if selected_payment != "All":

        filtered_data = filtered_data[
            filtered_data["Payment Method"] == selected_payment
        ]

    filtered_data = filtered_data[
        filtered_data["Churn Probability"]
        >= minimum_probability
    ]


    # ----------------------------------------------
    # Customer Search
    # ----------------------------------------------

    if customer_search.strip():

        search_value = customer_search.strip().lower()

        filtered_data = filtered_data[
            filtered_data["CustomerID"]
            .str.lower()
            .str.contains(
                search_value,
                na=False
            )
        ]


    # ----------------------------------------------
    # Sort Highest Risk First
    # ----------------------------------------------

    filtered_data = filtered_data.sort_values(
        "Churn Probability",
        ascending=False
    )


    # ----------------------------------------------
    # Filter Result Summary
    # ----------------------------------------------

    st.write(
        "Customers matching the selected filters: "
        f"**{len(filtered_data):,}**"
    )


    # ----------------------------------------------
    # Customer Table
    # ----------------------------------------------

    display_columns = [
        "CustomerID",
        "Churn Probability",
        "Risk Level",
        "Retention Flag",
        "Contract",
        "Tenure Months",
        "Internet Service",
        "Monthly Charges",
        "Total Charges",
        "Payment Method",
        "Dependents",
        "Online Security",
        "Tech Support",
        "Actual Churn"
    ]

    st.dataframe(
        filtered_data[display_columns],
        use_container_width=True,
        hide_index=True
    )


    # ----------------------------------------------
    # Download Filtered Customers
    # ----------------------------------------------

    download_csv = filtered_data[
        display_columns
    ].to_csv(
        index=False
    )

    st.download_button(
        label="Download Filtered Customers",
        data=download_csv,
        file_name="filtered_customer_churn_predictions.csv",
        mime="text/csv"
    )


# ==================================================
# TAB 3 — RETENTION STRATEGY
# ==================================================

with retention_tab:

    st.subheader("Retention Strategy")

    st.write(
        "These recommendations are hypotheses based on "
        "observed associations in the analysis. They should "
        "be validated through controlled experiments before "
        "being treated as causal interventions."
    )


    # ----------------------------------------------
    # Retention Strategy Table
    # ----------------------------------------------

    retention_strategy = pd.DataFrame(
        [
            [
                "Month-to-month contract",
                "Test contract-renewal incentives or longer-term plan options",
                "Compare retention against a control group"
            ],
            [
                "Short tenure",
                "Strengthen early-tenure onboarding and engagement",
                "Measure churn reduction during the first year"
            ],
            [
                "Fiber optic",
                "Investigate service experience and targeted support",
                "Use service records and controlled interventions"
            ],
            [
                "No online security",
                "Test security education, trials, or bundled options",
                "Compare adoption and subsequent retention"
            ],
            [
                "No technical support",
                "Consider proactive support or service-health outreach",
                "Measure retention against a control group"
            ],
            [
                "Electronic check",
                "Test alternative payment options or payment assistance",
                "Measure retention after payment-method intervention"
            ],
            [
                "High predicted churn",
                "Prioritize targeted retention outreach",
                "Run a controlled retention campaign"
            ]
        ],
        columns=[
            "Customer Signal",
            "Potential Retention Action",
            "Validation Approach"
        ]
    )


    st.dataframe(
        retention_strategy,
        use_container_width=True,
        hide_index=True
    )


    # ----------------------------------------------
    # Interpretation
    # ----------------------------------------------

    st.subheader("Important Interpretation")

    st.info(
        "The model identifies customers with higher predicted "
        "churn risk. It does not establish that any individual "
        "retention action will cause a customer to remain. "
        "Interventions should therefore be tested using "
        "controlled experiments such as A/B tests."
    )


# ==================================================
# TAB 4 — BUSINESS IMPACT
# ==================================================

with impact_tab:

    st.subheader(
        "Retention Campaign Scenario Analysis"
    )

    st.write(
        "This section estimates the potential financial impact "
        "of a hypothetical retention campaign using the "
        "held-out test-set results."
    )


    # ----------------------------------------------
    # Campaign Assumptions
    # ----------------------------------------------

    st.subheader("Campaign Assumptions")

    assumption_col1, assumption_col2, assumption_col3 = (
        st.columns(3)
    )

    with assumption_col1:

        st.metric(
            label="Customers Flagged",
            value=f"{high_risk_customers:,}"
        )

    with assumption_col2:

        st.metric(
            label="Actual Churners Flagged",
            value=f"{actual_churners_flagged:,}"
        )

    with assumption_col3:

        st.metric(
            label="Campaign Cost / Customer",
            value=f"{CAMPAIGN_COST_PER_CUSTOMER:,.0f}"
        )


    st.caption(
        "Illustrative retained-customer value: "
        f"{RETAINED_CUSTOMER_VALUE:,.0f} "
        "per retained customer."
    )


    # ----------------------------------------------
    # Break-Even Calculation
    # ----------------------------------------------

    total_campaign_cost = (
        high_risk_customers
        * CAMPAIGN_COST_PER_CUSTOMER
    )

    break_even_rate = (
        total_campaign_cost
        / (
            actual_churners_flagged
            * RETAINED_CUSTOMER_VALUE
        )
    )


    st.subheader("Break-Even Analysis")

    break_even_col1, break_even_col2 = st.columns(2)

    with break_even_col1:

        st.metric(
            label="Campaign Cost",
            value=f"{total_campaign_cost:,.0f}"
        )

    with break_even_col2:

        st.metric(
            label="Break-Even Retention Rate",
            value=f"{break_even_rate:.2%}"
        )


    st.info(
        "Under these illustrative assumptions, the campaign "
        "would need to retain approximately "
        f"**{break_even_rate:.2%}** of the flagged actual "
        "churners to recover the assumed campaign cost."
    )


    # ----------------------------------------------
    # Retention Scenario Analysis
    # ----------------------------------------------

    st.subheader("Retention Success Scenarios")

    retention_rates = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50
    ]

    scenario_rows = []

    for rate in retention_rates:

        estimated_retained = (
            actual_churners_flagged
            * rate
        )

        estimated_recovered_value = (
            estimated_retained
            * RETAINED_CUSTOMER_VALUE
        )

        estimated_net_benefit = (
            estimated_recovered_value
            - total_campaign_cost
        )

        scenario_rows.append(
            [
                rate,
                estimated_retained,
                total_campaign_cost,
                estimated_recovered_value,
                estimated_net_benefit
            ]
        )


    revenue_scenarios = pd.DataFrame(
        scenario_rows,
        columns=[
            "Retention Success Rate",
            "Estimated Retained Customers",
            "Campaign Cost",
            "Estimated Recovered Value",
            "Estimated Net Benefit"
        ]
    )


    # ----------------------------------------------
    # Format Scenario Table
    # ----------------------------------------------

    scenario_display = revenue_scenarios.copy()

    scenario_display[
        "Retention Success Rate"
    ] = (
        scenario_display["Retention Success Rate"]
        * 100
    ).map(
        lambda x: f"{x:.0f}%"
    )


    scenario_display[
        "Estimated Retained Customers"
    ] = (
        scenario_display[
            "Estimated Retained Customers"
        ]
        .round()
        .astype(int)
    )


    for column in [
        "Campaign Cost",
        "Estimated Recovered Value",
        "Estimated Net Benefit"
    ]:

        scenario_display[column] = (
            scenario_display[column]
            .round()
            .map(
                lambda x: f"{x:,.0f}"
            )
        )


    st.dataframe(
        scenario_display,
        use_container_width=True,
        hide_index=True
    )


    # ----------------------------------------------
    # Scenario Chart
    # ----------------------------------------------

    st.subheader(
        "Estimated Net Benefit by Retention Success Rate"
    )

    chart_data = revenue_scenarios[
        [
            "Retention Success Rate",
            "Estimated Net Benefit"
        ]
    ].copy()

    chart_data["Retention Success Rate"] = (
        chart_data["Retention Success Rate"]
        * 100
    )

    chart_data = chart_data.set_index(
        "Retention Success Rate"
    )

    st.line_chart(
        chart_data,
        y="Estimated Net Benefit"
    )


    st.caption(
        "These financial figures are illustrative scenario "
        "estimates, not realized or guaranteed financial results."
    )