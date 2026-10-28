"""
Module 3: Dual-Path AI Detective & Pydantic Guardrails (ai_detective.py)
VeriTriage AI — Autonomous Quality & Root-Cause Triage System
"""

import json
from typing import List, Literal, Dict, Any, Tuple
from pydantic import BaseModel, Field
from groq import Groq

EvidenceBadgeType = Literal[
    "STRONG EVIDENCE — Corroborated Across Lines/Logs",
    "WEAK EVIDENCE — Timing Coincidence Only",
    "CONTRADICTED — Ruled Out by Cross-Machine Control Data",
]


class HypothesisEvaluation(BaseModel):
    hypothesis_title: str = Field(
        ...,
        description="Name of the candidate root-cause hypothesis being evaluated."
    )
    suspect_category: Literal[
        "Internal Equipment / Maintenance",
        "External Supplier Material Lot",
        "Routine Operator / Shift Activity"
    ] = Field(
        ...,
        description="Category of the suspect root cause."
    )
    evidence_badge: EvidenceBadgeType = Field(
        ...,
        description="Strict qualitative evidence classification. Zero invented numerical confidence percentages allowed."
    )
    cited_log_ids: List[str] = Field(
        ...,
        description="Exact log IDs from the factory log table (e.g. ['#M-104', '#T-412', '#Q-88'])."
    )
    cross_machine_reasoning: str = Field(
        ...,
        description="Detailed causal reasoning integrating cross-chamber control data and statistical telemetry."
    )


class GovernedTriageOutput(BaseModel):
    case_id: str = Field(..., description="Excursion Case ID (e.g., EX-402 or EX-518).")
    primary_root_cause: str = Field(
        ...,
        description="Verified primary root cause supported by cross-machine control checks."
    )
    supplier_verdict: Literal[
        "EXONERATED — Protected from False Supplier Blame",
        "CONFIRMED DEFECTIVE — Multi-Chamber Material Contamination"
    ] = Field(
        ...,
        description="Fairness verdict on whether the external supplier lot caused the excursion."
    )
    recommended_quarantine_scope: str = Field(
        ...,
        description="Precision containment action specifying exact chambers/wafers to quarantine."
    )
    wafers_saved_from_false_scrap: int = Field(
        ...,
        description="Number of healthy wafers saved from unnecessary blanket quarantine."
    )
    hypotheses_board: List[HypothesisEvaluation] = Field(
        ...,
        description="Ranked evaluation of all candidate hypotheses with strict evidence badges."
    )
    eight_d_report: Dict[str, str] = Field(
        ...,
        description="Structured 8D / SCAR Triage Report covering sections D1 through D5."
    )


def _get_deterministic_fallback(
    case_id: str,
    case_json_str: str = "",
    stat_summary_str: str = ""
) -> Tuple[Dict[str, Any], GovernedTriageOutput]:
    """
    Returns rich, deterministic objects for both EX-402 and EX-518 so the application
    runs out-of-the-box without requiring an API key. Also inspects custom shift notes if added.
    """
    custom_note_present = "#CUST-" in case_json_str

    if case_id == "EX-402":
        naive_chatbot_output = {
            "headline_blame": "BLAME SUPPLIER: Apex Wafer Supply (Silicon Batch Lot #209)",
            "hallucinated_confidence": "94.7% Confidence (Invented Uncalibrated Probability)",
            "flawed_reasoning": (
                "New silicon wafer batch Lot #209 (#V-209) was loaded at 12:45 PM shortly before "
                "Feature_59 spiked above +3.0 sigma at 02:00 PM (#T-412). Because supplier material "
                "introduction preceded the excursion, Lot #209 is flagged as the root cause."
            ),
            "destructive_action": (
                "ISSUE IMMEDIATE SCAR TO APEX WAFER SUPPLY & BLANKET-QUARANTINE ALL 400 WAFERS "
                "ACROSS CHAMBERS #1, #2, AND #3 (Destroying 355 healthy wafers)."
            ),
            "failure_mode": (
                "Post Hoc Ergo Propter Hoc Fallacy — Blind to Cross-Chamber Control Log #Q-88 "
                "showing 300 wafers from Lot #209 passed with 0% defects on Chambers #1 and #3."
            ),
        }

        extra_cite = ["#M-104", "#T-412", "#Q-88"] + (["#CUST-1"] if custom_note_present else [])
        extra_note_text = (
            " (Incorporating live engineer shift note #CUST-1 from current triage session.)"
            if custom_note_present else ""
        )

        governed = GovernedTriageOutput(
            case_id="EX-402",
            primary_root_cause=(
                "Unscheduled pressure-valve reseal and manual throttle calibration drift on "
                "Chamber #2 (#M-104 at 01:35 PM) directly triggered the Chamber #2-only "
                "SECOM Feature_59 excursion (#T-412)." + extra_note_text
            ),
            supplier_verdict="EXONERATED — Protected from False Supplier Blame",
            recommended_quarantine_scope=(
                "Precision Quarantine: Hold ONLY the 45 wafers processed on Chamber #2 after "
                "01:35 PM maintenance ticket #M-104. Release 355 healthy wafers on Chambers #1 & #3."
            ),
            wafers_saved_from_false_scrap=355,
            hypotheses_board=[
                HypothesisEvaluation(
                    hypothesis_title="Chamber #2 Unscheduled Pressure-Valve Reseal & Throttle Drift",
                    suspect_category="Internal Equipment / Maintenance",
                    evidence_badge="STRONG EVIDENCE — Corroborated Across Lines/Logs",
                    cited_log_ids=extra_cite,
                    cross_machine_reasoning=(
                        "Maintenance ticket #M-104 modified Chamber #2's pressure valve and throttle calibration "
                        "at 01:35 PM, 25 minutes before SECOM Feature_59 breached +3.0 sigma exclusively on "
                        "Chamber #2 (#T-412, Wafer Runs #422-#435). Cross-line check #Q-88 confirms the "
                        "anomaly is physically isolated to Chamber #2."
                    ),
                ),
                HypothesisEvaluation(
                    hypothesis_title="Apex Wafer Supply — Silicon Batch Lot #209 Material Defect",
                    suspect_category="External Supplier Material Lot",
                    evidence_badge="CONTRADICTED — Ruled Out by Cross-Machine Control Data",
                    cited_log_ids=["#V-209", "#Q-88", "#T-412"],
                    cross_machine_reasoning=(
                        "Although Lot #209 was loaded at 12:45 PM (#V-209), it was distributed across "
                        "Chambers #1, #2, and #3 simultaneously. Log #Q-88 proves Chambers #1 and #3 "
                        "processed 300 wafers from Lot #209 with a 100% pass rate (0 failures). "
                        "A material lot defect cannot selectively fail only in Chamber #2."
                    ),
                ),
                HypothesisEvaluation(
                    hypothesis_title="Chamber #1 Routine Optical Viewport Wipe",
                    suspect_category="Routine Operator / Shift Activity",
                    evidence_badge="WEAK EVIDENCE — Timing Coincidence Only",
                    cited_log_ids=["#M-101", "#Q-88"],
                    cross_machine_reasoning=(
                        "Log #M-101 occurred on Chamber #1 at 01:15 PM, and Chamber #1 exhibited "
                        "zero telemetry drift and 100% yield (#Q-88)."
                    ),
                ),
            ],
            eight_d_report={
                "D1_Team_And_Telemetry_Scope": (
                    f"Fab Yield Triage Team | Case EX-402 | Sensor: SECOM Feature_59 | {stat_summary_str}"
                ),
                "D2_Problem_Description": (
                    "At 02:00 PM (#T-412), SECOM Feature_59 exceeded the Bonferroni-corrected statistical "
                    "control limit exclusively on Chamber #2 across Wafer Runs #422-#435 (14 electrical gate failures)."
                ),
                "D3_Interim_Containment_Action": (
                    "PRECISION CONTAINMENT: Quarantine ONLY the 45 wafers processed on Chamber #2 post-01:35 PM. "
                    "Do NOT quarantine Chambers #1 and #3 (saving 355 healthy wafers from Lot #209). "
                    "Do NOT issue a false SCAR to Apex Wafer Supply."
                ),
                "D4_Root_Cause_And_Escape_Point": (
                    "Root Cause: Unscheduled pressure-valve reseal & manual throttle calibration on Chamber #2 (#M-104) "
                    "without post-maintenance qualification wafer check. Supplier Lot #209 is mathematically "
                    "exonerated by cross-chamber control data (#Q-88: 300/300 passing wafers on Chambers #1 & #3)."
                ),
                "D5_Permanent_Corrective_Action": (
                    "1. Recalibrate Chamber #2 mass-flow throttle valve against golden reference standard.\n"
                    "2. Enforce mandatory 3-wafer SPC qualification check after any pressure-valve intervention (#M-104)."
                ),
            },
        )
        return naive_chatbot_output, governed

    # Case B: EX-518 (Confirmed Multi-Chamber Supplier Contamination)
    naive_chatbot_output = {
        "headline_blame": "BLAME SUPPLIER: ChemCorp International (Precursor Batch #88)",
        "hallucinated_confidence": "96.2% Confidence (Invented Uncalibrated Probability)",
        "flawed_reasoning": (
            "ChemCorp Precursor Batch #88 arrived at 08:30 AM (#V-312) before Feature_103 spiked at "
            "10:30 AM (#T-518). Flagging supplier based on timestamp proximity without verifying "
            "whether Chamber #4 control line or maintenance logs corroborate the claim."
        ),
        "destructive_action": (
            "QUARANTINE ALL 400 WAFERS (Happens to match required quarantine here, but lacks "
            "cross-chamber control proof #Q-94 and invents fake 96.2% confidence score)."
        ),
        "failure_mode": (
            "Right conclusion by accident — relies on unverified timestamp correlation instead of "
            "multi-chamber manifold replication (#L-405, #T-518) vs. Chamber #4 control batch (#Q-94)."
        ),
    }

    extra_cite = ["#V-312", "#L-405", "#T-518", "#Q-94"] + (["#CUST-1"] if custom_note_present else [])
    extra_note_text = (
        " (Incorporating live engineer shift note #CUST-1 from current triage session.)"
        if custom_note_present else ""
    )

    governed = GovernedTriageOutput(
        case_id="EX-518",
        primary_root_cause=(
            "Confirmed chemical precursor contamination in ChemCorp International Dielectric Precursor "
            "Batch #88 (#V-312, #L-405), producing simultaneous SECOM Feature_103 > +3.0 sigma spikes "
            "across all 3 connected tools (Chambers #1, #2, and #3, #T-518) while control Chamber #4 "
            "on legacy Batch #81 remained at 0.0 sigma (#Q-94)." + extra_note_text
        ),
        supplier_verdict="CONFIRMED DEFECTIVE — Multi-Chamber Material Contamination",
        recommended_quarantine_scope=(
            "Full Lot Quarantine: Quarantine all 400 wafers processed across Chambers #1, #2, and #3 "
            "on ChemCorp Batch #88, isolate central manifold (#L-405), and issue evidence-backed SCAR."
        ),
        wafers_saved_from_false_scrap=0,
        hypotheses_board=[
            HypothesisEvaluation(
                hypothesis_title="ChemCorp International — Dielectric Precursor Batch #88 Contamination",
                suspect_category="External Supplier Material Lot",
                evidence_badge="STRONG EVIDENCE — Corroborated Across Lines/Logs",
                cited_log_ids=extra_cite,
                cross_machine_reasoning=(
                    "Precursor Batch #88 was manifolded to Chambers #1, #2, and #3 at 09:15 AM (#L-405). "
                    "At 10:30 AM, SECOM Feature_103 breached +3.0 sigma simultaneously on all three chambers (#T-518). "
                    "Meanwhile, Control Chamber #4 running legacy Batch #81 showed 0.0 sigma drift and 100% yield (#Q-94). "
                    "Multi-tool replication + negative control chamber proves supplier batch contamination."
                ),
            ),
            HypothesisEvaluation(
                hypothesis_title="Internal Mechanical Maintenance or Valve Calibration Drift",
                suspect_category="Internal Equipment / Maintenance",
                evidence_badge="CONTRADICTED — Ruled Out by Cross-Machine Control Data",
                cited_log_ids=["#M-210", "#T-518"],
                cross_machine_reasoning=(
                    "Shift log #M-210 confirms zero mechanical maintenance, valve adjustments, or calibration "
                    "changes occurred on any chamber today, and independent mechanical drift would not occur "
                    "simultaneously at 10:30 AM across three separate chambers."
                ),
            ),
            HypothesisEvaluation(
                hypothesis_title="Ambient Cleanroom / Facility Power Fluctuation",
                suspect_category="Routine Operator / Shift Activity",
                evidence_badge="CONTRADICTED — Ruled Out by Cross-Machine Control Data",
                cited_log_ids=["#Q-94", "#T-518"],
                cross_machine_reasoning=(
                    "A facility-wide cleanroom or power fluctuation would also impact Chamber #4, "
                    "which operated in the same bay with 0.0 sigma drift and 100% passing wafers (#Q-94)."
                ),
            ),
        ],
        eight_d_report={
            "D1_Team_And_Telemetry_Scope": (
                f"Fab Yield & Supplier Quality Team | Case EX-518 | Sensor: SECOM Feature_103 | {stat_summary_str}"
            ),
            "D2_Problem_Description": (
                "At 10:30 AM (#T-518), SECOM Feature_103 spiked > +3.0 sigma simultaneously across "
                "Chambers #1, #2, and #3, causing dielectric film thickness failures across Wafer Runs #422-#435."
            ),
            "D3_Interim_Containment_Action": (
                "FULL LOT CONTAINMENT: Quarantine all 400 wafers exposed to ChemCorp Precursor Batch #88 "
                "across Chambers #1, #2, and #3. Purge shared precursor manifold (#L-405) and switch to verified reserve lot."
            ),
            "D4_Root_Cause_And_Escape_Point": (
                "Root Cause: Out-of-spec Dielectric Precursor Batch #88 from ChemCorp International (#V-312). "
                "Confirmed via positive replication across Chambers #1-#3 (#T-518), zero internal maintenance (#M-210), "
                "and negative control verification on Chamber #4 running legacy Batch #81 (#Q-94)."
            ),
            "D5_Permanent_Corrective_Action": (
                "1. Issue formal SCAR to ChemCorp International with cross-chamber telemetry proof (#T-518, #Q-94).\n"
                "2. Require incoming Certificate of Analysis (CoA) trace-moisture spectroscopy on all precursor lots prior to manifold hookup."
            ),
        },
    )
    return naive_chatbot_output, governed


def run_dual_investigation(
    case_id: str,
    case_json_str: str,
    stat_summary_str: str,
    api_key: str = "",
    model_name: str = "llama-3.3-70b-versatile"
) -> Dict[str, Any]:
    """
    Runs the Dual-Path Investigation:
    1) Standard Ungoverned AI Simulation (demonstrating post-hoc timing bias & invented % confidence)
    2) VeriTriage AI Governed Detective Board (enforcing cross-machine control checks & Pydantic schema)
    Uses deterministic fallback when `api_key` is empty or if API call fails.
    """
    naive_fallback, governed_fallback = _get_deterministic_fallback(
        case_id=case_id,
        case_json_str=case_json_str,
        stat_summary_str=stat_summary_str,
    )

    clean_key = (api_key or "").strip()
    if not clean_key:
        return {
            "mode": "Deterministic Offline Engine (Zero-Key Ready)",
            "naive_output": naive_fallback,
            "governed_output": governed_fallback,
            "error": None,
        }

    schema_str = json.dumps(GovernedTriageOutput.model_json_schema(), indent=2)
    system_prompt = (
        "You are VeriTriage AI, a Staff Semiconductor Yield & Quality Triage Detective.\n"
        "CRITICAL GOVERNANCE RULES:\n"
        "1. NEVER invent numerical confidence percentages (e.g., never output '92% confidence').\n"
        "2. You MUST restrict every hypothesis `evidence_badge` to one of the exact 3 literal values in the schema:\n"
        "   - 'STRONG EVIDENCE — Corroborated Across Lines/Logs'\n"
        "   - 'WEAK EVIDENCE — Timing Coincidence Only'\n"
        "   - 'CONTRADICTED — Ruled Out by Cross-Machine Control Data'\n"
        "3. Always perform a Cross-Machine Control Check: if a supplier lot is loaded on multiple chambers "
        "and only one chamber fails while other chambers pass 100%, you MUST exonerate the supplier and "
        "classify the supplier hypothesis as 'CONTRADICTED — Ruled Out by Cross-Machine Control Data'.\n"
        "4. Conversely, if a supplier batch causes simultaneous excursions across multiple chambers while "
        "an independent control chamber on an older batch passes 100% with zero maintenance performed, "
        "confirm the supplier defect with 'STRONG EVIDENCE — Corroborated Across Lines/Logs'.\n"
        "5. Cite exact log IDs (e.g., #V-209, #M-104, #T-412, #Q-88) in `cited_log_ids`.\n"
        "6. Respond ONLY with a valid JSON object matching this exact Pydantic JSON Schema:\n"
        f"{schema_str}"
    )

    user_prompt = (
        f"Investigate Semiconductor Excursion Case ID: {case_id}\n\n"
        f"LOCAL SCIPY STATISTICAL PRE-PROCESSOR SUMMARY:\n{stat_summary_str}\n\n"
        f"FACTORY LOGS & SCENARIO DATA (JSON):\n{case_json_str}\n\n"
        "Return ONLY valid JSON conforming strictly to GovernedTriageOutput."
    )

    try:
        client = Groq(api_key=clean_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw_json_text = completion.choices[0].message.content
        validated_output = GovernedTriageOutput.model_validate_json(raw_json_text)
        return {
            "mode": f"Live Groq LLM ({model_name}) + Pydantic Guardrails",
            "naive_output": naive_fallback,
            "governed_output": validated_output,
            "error": None,
        }
    except Exception as exc:
        return {
            "mode": f"Deterministic Fallback (Groq API fallback triggered: {type(exc).__name__})",
            "naive_output": naive_fallback,
            "governed_output": governed_fallback,
            "error": str(exc),
        }
