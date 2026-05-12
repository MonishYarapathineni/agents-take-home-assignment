import Anthropic from "@anthropic-ai/sdk";
import {
  create_task,
  draft_message,
  escalate,
  find_slots,
  getToolCallsForItem,
  hold_slot,
  lookup_policy,
  search_patient,
  verify_insurance,
  withItemContext,
} from "./tools.js";
import type { InboxItem, ItemOutput } from "./types.js";

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// ─── LLM-based classification + extraction ───────────────────────────────────
//
// One LLM call per item. Returns everything the handlers need:
// - classification signal (what kind of item is this)
// - extracted intake fields (who, what, insurance, contact)
//
// All natural language understanding lives here. Downstream handlers are
// pure orchestration logic — they trust this output and call tools accordingly.

type ClassificationSignal =
  | "safeguarding"
  | "same_day_cancellation"
  | "incomplete_referral"
  | "clinical_question"
  | "new_referral"
  | "other";

interface ItemAnalysis {
  signal: ClassificationSignal;
  language: "en" | "es";
  child_name: string | null;
  dob_or_age: string | null;
  parent_name: string | null;
  parent_phone: string | null;
  parent_email: string | null;
  discipline: ("SLP" | "OT" | "PT")[] | null;
  diagnosis_or_concern: string | null;
  payer: string | null;
  member_id: string | null;
  preferences: string | null;
  safeguarding_summary: string | null; // non-null only when signal === "safeguarding"
}

async function analyzeItem(item: InboxItem): Promise<ItemAnalysis> {
  const fallback: ItemAnalysis = {
    signal: "other",
    language: "en",
    child_name: null,
    dob_or_age: null,
    parent_name: null,
    parent_phone: null,
    parent_email: null,
    discipline: null,
    diagnosis_or_concern: null,
    payer: null,
    member_id: null,
    preferences: null,
    safeguarding_summary: null,
  };

  try {
    const response = await anthropic.messages.create({
      model: "claude-sonnet-4-5",
      max_tokens: 1024,
      tools: [
        {
          name: "triage_item",
          description: "Classify and extract intake fields from a pediatric therapy clinic inbox message.",
          input_schema: {
            type: "object" as const,
            properties: {
              signal: {
                type: "string",
                enum: ["safeguarding", "same_day_cancellation", "incomplete_referral", "clinical_question", "new_referral", "other"],
                description: "Classification signal. safeguarding takes priority over all others — use it if there is any hint of harm, abuse, neglect, or unsafe caregiving, even alongside a referral. same_day_cancellation: cancelling/rescheduling today. incomplete_referral: referral with critical fields explicitly missing. clinical_question: developmental question with no referral data. new_referral: intake request with enough data to process.",
              },
              language: {
                type: "string",
                enum: ["en", "es"],
                description: "Primary language of the message.",
              },
              child_name: { type: ["string", "null"], description: "Full name of the child — include both first AND last name if both are present in the message. Never return first name only if a last name is available." },
              dob_or_age: { type: ["string", "null"], description: "Date of birth as YYYY-MM-DD or age string like 'age 6'." },
              parent_name: { type: ["string", "null"], description: "Name of parent or guardian." },
              parent_phone: { type: ["string", "null"], description: "Parent phone number." },
              parent_email: { type: ["string", "null"], description: "Parent email address." },
              discipline: {
                oneOf: [
                  { type: "null" },
                  { type: "array", items: { type: "string", enum: ["SLP", "OT", "PT"] }, minItems: 1 },
                ],
                description: "Therapy disciplines requested.",
              },
              diagnosis_or_concern: { type: ["string", "null"], description: "Clinical concern or reason for referral." },
              payer: { type: ["string", "null"], description: "Insurance company name as stated." },
              member_id: { type: ["string", "null"], description: "Insurance member ID." },
              preferences: { type: ["string", "null"], description: "Scheduling preferences if mentioned." },
              safeguarding_summary: { type: ["string", "null"], description: "If signal is safeguarding: brief factual description of the concern for internal staff only. Otherwise null." },
            },
            required: ["signal", "language", "child_name", "dob_or_age", "parent_name", "parent_phone", "parent_email", "discipline", "diagnosis_or_concern", "payer", "member_id", "preferences", "safeguarding_summary"],
          },
        },
      ],
      tool_choice: { type: "tool" as const, name: "triage_item" },
      temperature: 0,
      messages: [
        {
          role: "user",
          content: `You are a triage assistant for Cedar Kids Therapy, a pediatric therapy practice.
Analyze this inbox message and call the triage_item tool with your analysis.

Message:
Channel: ${item.channel}
Sender: ${item.sender}
Subject: ${item.subject}
Body: ${item.body}

Rules:
- safeguarding always takes priority — any hint of harm, abuse, or unsafe caregiving overrides other signals
- Extract fields from the full message including sender metadata, not just the body
- For incomplete_referral: only when structured like a referral but missing DOB, parent contact, or insurance`,
        },
      ],
    });

    const toolUse = response.content.find((b) => b.type === "tool_use");
    if (!toolUse || toolUse.type !== "tool_use") return fallback;
    // Merge with fallback to guarantee all keys present even if model omits null fields
    return { ...fallback, ...(toolUse.input as Partial<ItemAnalysis>) } as ItemAnalysis;
  } catch (err) {
    console.warn(`[analyzeItem] failed for ${item.id}:`, err);
    return fallback;
  }
}

// ─── Draft generation ────────────────────────────────────────────────────────

interface DraftContext {
  channel: string;
  language: "en" | "es";
  recipientName: string;
  childName: string | null;
  situation: string;
  constraints: string[];
  fallback: string;
}

async function generateDraft(ctx: DraftContext): Promise<string> {
  const constraintBlock = ctx.constraints.map((c) => `- ${c}`).join("\n");
  const prompt = [
    `You are drafting a brief, professional reply on behalf of Cedar Kids Therapy, a pediatric therapy practice.`,
    `Channel: ${ctx.channel}. Language: ${ctx.language === "es" ? "Spanish" : "English"}.`,
    `Recipient: ${ctx.recipientName}. Child: ${ctx.childName ?? "unknown"}.`,
    `Situation: ${ctx.situation}`,
    `\nConstraints:`,
    constraintBlock,
    `\nFormat rules — follow exactly:`,
    `- Exactly 2 sentences. No more.`,
    `- Sentence 1: factual acknowledgement only. Do NOT add enthusiasm, compliments, or emotional amplification. Do not say "we'd love to help", "so glad you reached out", "wonderful", or similar filler.`,
    `- Sentence 2: one concrete operational next step. State who will contact them or what happens next. Be specific.`,
    `- Start with "Hi [name]" or Spanish equivalent. Do not start with "I".`,
    `- No sign-off, no closing phrase, no name, no "warm regards", no "Cedar Kids Therapy" at the end.`,
    `- Do not provide clinical advice. Do not imply the message has already been sent.`,
    `\nGood example: "Hi Rachel, we received Owen's referral and have flagged a billing question about your insurance plan. Our billing team will contact you within one business day to discuss your options."`,
    `Bad example: "Hi Rachel, thank you so much for reaching out — we're so glad you did and would love to support Owen on his journey! Our wonderful billing team will be in touch soon."`,
    ctx.language === "es" ? "\nWrite entirely in Spanish." : "",
    `\nReturn only the 2-sentence message body. Nothing else.`,
  ].filter(Boolean).join("\n");

  try {
    const response = await anthropic.messages.create({
      model: "claude-sonnet-4-5",
      max_tokens: 120,
      temperature: 0.3,
      messages: [{ role: "user", content: prompt }],
    });
    const text = response.content.find((b) => b.type === "text");
    return text?.text?.trim() ?? ctx.fallback;
  } catch (err) {
    console.warn("[generateDraft] API call failed, using fallback:", err);
    return ctx.fallback;
  }
}

// ─── Entry point ─────────────────────────────────────────────────────────────

export async function runAgent(inbox: InboxItem[]): Promise<ItemOutput[]> {
  const results = await Promise.allSettled(
    inbox.map((item) => triageItem(item)),
  );

  return results.map((result, index) => {
    if (result.status === "fulfilled") return result.value;
    console.error(`[agent] item ${inbox[index].id} failed:`, result.reason);
    return fallbackOutput(inbox[index], String(result.reason));
  });
}

// ─── Per-item orchestrator ────────────────────────────────────────────────────

async function triageItem(item: InboxItem): Promise<ItemOutput> {
  return withItemContext(item.id, async () => {
    const analysis = await analyzeItem(item);

    switch (analysis.signal) {
      case "safeguarding":         return handleSafeguarding(item, analysis);
      case "same_day_cancellation": return handleSameDayCancellation(item, analysis);
      case "incomplete_referral":  return handleIncompleteReferral(item, analysis);
      case "clinical_question":    return handleClinicalQuestion(item, analysis);
      case "new_referral":         return handleNewReferral(item, analysis);
      default:                     return handleOther(item, analysis);
    }
  });
}

// ─── Handlers ────────────────────────────────────────────────────────────────

async function handleSafeguarding(item: InboxItem, analysis: ItemAnalysis): Promise<ItemOutput> {
  await lookup_policy({ topic: "safeguarding" });

  const childName = analysis.child_name ?? "unknown child";
  const parentName = analysis.parent_name ?? "the parent";
  const parentPhone = analysis.parent_phone ?? "unknown";

  const esc = await escalate({
    item_id: item.id,
    reason: analysis.safeguarding_summary ??
      `Possible safeguarding concern disclosed in ${item.channel} from ${item.sender}. Requires clinical lead review.`,
    severity: "P0",
  });

  const task = await create_task({
    assignee: "clinical_lead",
    title: `SAFEGUARDING — Same-hour review required: ${childName}`,
    due: todayIso(),
    notes: `${analysis.safeguarding_summary ?? "Safeguarding concern disclosed."} Contact: ${parentName}, ${parentPhone}. Clinical lead must determine mandated-reporter obligations before any outbound contact. Do not include any details about the concern in outbound messages.`,
  });

  const draftBody = await generateDraft({
    channel: item.channel,
    language: analysis.language,
    recipientName: analysis.parent_name ?? "there",
    childName: analysis.child_name,
    situation: "A parent reached out about their child. A safeguarding concern has been flagged internally and escalated to the clinical lead. This reply must be a neutral acknowledgement only — no reference to the concern.",
    constraints: [
      "Do not mention safeguarding, abuse, harm, or any details from the message",
      "Do not provide investigative advice",
      "Tell them a team member will be in touch shortly",
      "Do not confirm or imply any clinical services are scheduled",
    ],
    fallback: `Hi ${analysis.parent_name ?? "there"}, thank you for reaching out about ${analysis.child_name ?? "your child"}. A member of our team will be in touch shortly to discuss next steps.`,
  });

  const draft = await draft_message({
    recipient: analysis.parent_phone ?? analysis.parent_email ?? item.sender,
    channel: inferChannel(item),
    body: draftBody,
    language: analysis.language,
  });

  return {
    item_id: item.id,
    classification: "safeguarding",
    urgency: "P0",
    requires_human_review: true,
    extracted_intake: buildExtractedIntake(analysis),
    missing_info: buildMissingInfo(analysis),
    tools_called: getToolCallsForItem(item.id),
    recommended_next_action:
      "Clinical lead must review safeguarding disclosure within the hour and determine mandated-reporter obligations before any scheduling or outbound clinical contact.",
    draft_reply: draftBody,
    task_ids: [task.data.task_id],
    escalation: { reason: esc.args.reason as string, severity: "P0" },
    decision_rationale:
      `Safeguarding signal detected: ${analysis.safeguarding_summary ?? "concern disclosed in message"}. P0 per policy regardless of any concurrent referral request. Neutral acknowledgement drafted — no clinical or investigative content. Clinical lead task created for same-hour review.`,
  };
}

async function handleSameDayCancellation(item: InboxItem, analysis: ItemAnalysis): Promise<ItemOutput> {
  await lookup_policy({ topic: "cancellation" });

  // Try to find existing patient record
  const patientSearch = analysis.child_name
    ? await search_patient({
        name: analysis.child_name,
        dob: analysis.dob_or_age?.match(/^\d{4}-\d{2}-\d{2}$/)
          ? analysis.dob_or_age
          : undefined,
      })
    : null;

  const patientFound = (patientSearch?.data.length ?? 0) > 0;
  const patientId = patientFound ? patientSearch!.data[0].patient_id : null;

  const slots = await find_slots({
    discipline: analysis.discipline?.[0],
  });
  const firstSlot = slots.data[0];

  let holdId: string | null = null;
  if (firstSlot) {
    const hold = await hold_slot({
      slot_id: firstSlot.slot_id,
      patient_ref: patientId ?? `${analysis.child_name ?? "unknown"} (unconfirmed)`,
    });
    holdId = hold.data.hold_id;
  }

  const task = await create_task({
    assignee: "front_desk",
    title: `Same-day cancellation — ${analysis.child_name ?? "unknown patient"}`,
    due: todayIso(),
    notes: `${analysis.parent_name ?? "Parent"} (${analysis.parent_phone ?? analysis.parent_email ?? "unknown contact"}) requesting same-day reschedule for ${analysis.child_name ?? "patient"}. ${firstSlot ? `Slot ${firstSlot.slot_id} (${firstSlot.start}) held as ${holdId} pending staff confirmation.` : "No matching slots found — staff to check manually."} Patient record: ${patientId ?? "not found, verify manually"}.`,
  });

  const draftBody = await generateDraft({
    channel: item.channel,
    language: analysis.language,
    recipientName: analysis.parent_name ?? "there",
    childName: analysis.child_name,
    situation: `A parent has requested to cancel or reschedule a same-day appointment for ${analysis.child_name ?? "their child"}. The cancellation has been noted and a slot has been held pending staff review.`,
    constraints: [
      "Do not schedule or confirm any appointment",
      "Tell them a staff member will be in touch to confirm a new time",
      "Be warm and brief",
    ],
    fallback: `Hi ${analysis.parent_name ?? "there"}, we've noted the cancellation and a staff member will be in touch shortly to arrange a new time.`,
  });

  await draft_message({
    recipient: analysis.parent_email ?? analysis.parent_phone ?? item.sender,
    channel: inferChannel(item),
    body: draftBody,
    language: analysis.language,
  });

  return {
    item_id: item.id,
    classification: "scheduling",
    urgency: "P1",
    requires_human_review: true,
    extracted_intake: buildExtractedIntake(analysis),
    missing_info: [],
    tools_called: getToolCallsForItem(item.id),
    recommended_next_action: holdId
      ? `Front desk should confirm held slot ${holdId} (${firstSlot?.start}) with parent and finalize reschedule. Hold expires in 30 minutes.`
      : "Front desk should contact parent to reschedule — no slots available, check manually.",
    draft_reply: draftBody,
    task_ids: [task.data.task_id],
    escalation: null,
    decision_rationale:
      `Same-day cancellation is P1 per scheduling policy. ${patientFound ? `Patient found: ${patientId}.` : "Patient not found in system — verify manually."} ${holdId ? `Slot held (${firstSlot?.start}) pending staff confirmation.` : "No matching slot found."}`,
  };
}

async function handleIncompleteReferral(item: InboxItem, analysis: ItemAnalysis): Promise<ItemOutput> {
  await lookup_policy({ topic: "service_lines" });

  const missingInfo = buildMissingInfo(analysis);

  const task = await create_task({
    assignee: "intake",
    title: `Incomplete referral — ${analysis.child_name ?? "unknown child"} (${item.sender})`,
    due: todayIso(),
    notes: `Referral received from ${item.sender} is missing: ${missingInfo.join(", ")}. Contact referring provider to request complete information before scheduling.`,
  });

  const draftBody = await generateDraft({
    channel: item.channel,
    language: analysis.language,
    recipientName: item.sender,
    childName: analysis.child_name,
    situation: `A referral was received for ${analysis.child_name ?? "a child"} but is missing required fields: ${missingInfo.join(", ")}. This message goes to the referring provider asking them to resend with complete information.`,
    constraints: [
      "This is a message to the referring provider, not the family",
      `Ask them to resend with the missing fields: ${missingInfo.join(", ")}`,
      "Be professional and concise",
    ],
    fallback: `Hi, this is Cedar Kids Therapy following up on the referral for ${analysis.child_name ?? "your patient"}. We are missing the following information: ${missingInfo.join(", ")}. Could you please resend with those details? Thank you.`,
  });

  await draft_message({
    recipient: item.sender,
    channel: "phone",
    body: draftBody,
    language: analysis.language,
  });

  return {
    item_id: item.id,
    classification: "missing_paperwork",
    urgency: "P2",
    requires_human_review: true,
    extracted_intake: buildExtractedIntake(analysis),
    missing_info: missingInfo,
    tools_called: getToolCallsForItem(item.id),
    recommended_next_action:
      `Intake should contact ${item.sender} to request missing fields before this referral can be processed.`,
    draft_reply: draftBody,
    task_ids: [task.data.task_id],
    escalation: null,
    decision_rationale:
      `Referral from ${item.sender} is missing required fields: ${missingInfo.join(", ")}. Cannot verify insurance or contact family without this data.`,
  };
}

async function handleClinicalQuestion(item: InboxItem, analysis: ItemAnalysis): Promise<ItemOutput> {
  await lookup_policy({ topic: "clinical_advice" });

  const draftBody = await generateDraft({
    channel: item.channel,
    language: analysis.language,
    recipientName: analysis.parent_name ?? "there",
    childName: analysis.child_name,
    situation: `A parent sent a clinical or developmental question about ${analysis.child_name ?? "their child"}. Policy does not allow clinical advice in messages — route to an SLP screening or evaluation.`,
    constraints: [
      "Do not answer the clinical question",
      "Acknowledge the question warmly",
      "Offer to connect them with a clinician via a screening call or evaluation",
    ],
    fallback: `Hi ${analysis.parent_name ?? "there"}, thank you for your question. Our team isn't able to provide clinical guidance over messages, but one of our therapists would be happy to help during a screening call or evaluation. Please reach out if you'd like to set something up.`,
  });

  await draft_message({
    recipient: analysis.parent_email ?? analysis.parent_phone ?? item.sender,
    channel: inferChannel(item),
    body: draftBody,
    language: analysis.language,
  });

  return {
    item_id: item.id,
    classification: "clinical_question",
    urgency: "P2",
    requires_human_review: true,
    extracted_intake: buildExtractedIntake(analysis),
    missing_info: buildMissingInfo(analysis),
    tools_called: getToolCallsForItem(item.id),
    recommended_next_action:
      "No further action required beyond draft reply. If parent responds affirmatively, route to screening intake.",
    draft_reply: draftBody,
    task_ids: [],
    escalation: null,
    decision_rationale:
      "Parent sent a clinical or developmental question. Policy prohibits clinical advice from automated systems. Draft routes to clinician screening — no clinical content included.",
  };
}

async function handleNewReferral(item: InboxItem, analysis: ItemAnalysis): Promise<ItemOutput> {
  // Search for existing patient if we have enough to go on
  let patientId: string | null = null;
  if (analysis.child_name) {
    const dobArg = analysis.dob_or_age?.match(/^\d{4}-\d{2}-\d{2}$/)
      ? analysis.dob_or_age
      : undefined;
    const patient = await search_patient({
      name: analysis.child_name,
      dob: dobArg,
    });
    if (patient.data.length > 0) {
      patientId = patient.data[0].patient_id;
    }
  }

  const insurance = await verify_insurance({
    payer: analysis.payer ?? undefined,
    member_id: analysis.member_id ?? undefined,
  });

  // Policy: billing system supersedes referral doc — surface discrepancy if present
  const discrepancy =
    analysis.payer &&
    insurance.data.plan &&
    insurance.data.plan.toLowerCase() !== analysis.payer.toLowerCase()
      ? `Billing system payer (${insurance.data.plan}) differs from referral (${analysis.payer}) — billing system is source of truth per policy.`
      : null;

  const isOutOfNetwork = insurance.data.status === "out_of_network";
  const isExpired = insurance.data.status === "expired";
  const isUnknownInsurance = insurance.data.status === "unknown" || !analysis.payer;

  // Expired — billing must verify before any slot hold
  if (isExpired) {
    await lookup_policy({ topic: "insurance" });

    const task = await create_task({
      assignee: "billing",
      title: `Expired insurance — ${analysis.child_name ?? "unknown child"}`,
      due: tomorrowIso(),
      notes: `${analysis.payer} returned expired status. ${discrepancy ?? ""} Parent: ${analysis.parent_name ?? "unknown"}, ${analysis.parent_phone ?? analysis.parent_email ?? "unknown contact"}. Verify active coverage before proceeding.`,
    });

    const draftBody = await generateDraft({
      channel: item.channel,
      language: analysis.language,
      recipientName: analysis.parent_name ?? "there",
      childName: analysis.child_name,
      situation: `Referral received for ${analysis.child_name ?? "the child"} but insurance (${analysis.payer}) appears expired. Billing needs to verify coverage before scheduling.`,
      constraints: [
        "Do not schedule or hold any appointment",
        "Tell the parent billing will follow up to verify coverage",
        "Be warm — this is not a rejection",
      ],
      fallback: `Hi ${analysis.parent_name ?? "there"}, thank you for the referral. Our billing team needs to verify the current insurance coverage before we move forward. Someone will be in touch shortly.`,
    });

    await draft_message({
      recipient: analysis.parent_email ?? analysis.parent_phone ?? item.sender,
      channel: inferChannel(item),
      body: draftBody,
      language: analysis.language,
    });

    return {
      item_id: item.id,
      classification: "new_referral",
      urgency: "P2",
      requires_human_review: true,
      extracted_intake: buildExtractedIntake(analysis),
      missing_info: [
        "active insurance coverage — billing system shows expired",
        ...(discrepancy ? [discrepancy] : []),
      ],
      tools_called: getToolCallsForItem(item.id),
      recommended_next_action: "Billing must verify active coverage before any slot hold or scheduling.",
      draft_reply: draftBody,
      task_ids: [task.data.task_id],
      escalation: null,
      decision_rationale: `Insurance verification returned expired for ${analysis.payer}. ${discrepancy ?? ""} No slot held. Billing task created.`,
    };
  }

  // Out of network — benefits conversation required before slot hold
  if (isOutOfNetwork) {
    await lookup_policy({ topic: "insurance" });

    const task = await create_task({
      assignee: "billing",
      title: `Out-of-network benefits review — ${analysis.child_name ?? "unknown child"}`,
      due: tomorrowIso(),
      notes: `${analysis.payer} verified out of network. ${discrepancy ?? ""} Parent: ${analysis.parent_name ?? "unknown"}, ${analysis.parent_phone ?? analysis.parent_email ?? "unknown contact"}. Benefits conversation required before slot hold.`,
    });

    const draftBody = await generateDraft({
      channel: item.channel,
      language: analysis.language,
      recipientName: analysis.parent_name ?? "there",
      childName: analysis.child_name,
      situation: `Referral received for ${analysis.child_name ?? "the child"} but insurance (${analysis.payer}) is out of network. Billing needs to discuss options with the family before scheduling.`,
      constraints: [
        "Do not schedule or hold any appointment",
        "Tell the parent billing will follow up to discuss options",
        "Be warm and reassuring — this is not a rejection",
      ],
      fallback: `Hi ${analysis.parent_name ?? "there"}, thank you for the referral. Our billing team needs to review the ${analysis.payer} plan before we move forward. Someone will be in touch with your options.`,
    });

    await draft_message({
      recipient: analysis.parent_email ?? analysis.parent_phone ?? item.sender,
      channel: inferChannel(item),
      body: draftBody,
      language: analysis.language,
    });

    return {
      item_id: item.id,
      classification: "new_referral",
      urgency: "P2",
      requires_human_review: true,
      extracted_intake: buildExtractedIntake(analysis),
      missing_info: discrepancy ? [discrepancy] : [],
      tools_called: getToolCallsForItem(item.id),
      recommended_next_action: "Billing must complete benefits conversation before any slot hold or scheduling.",
      draft_reply: draftBody,
      task_ids: [task.data.task_id],
      escalation: null,
      decision_rationale: `Insurance verified out-of-network (${analysis.payer}). ${discrepancy ?? ""} Policy requires benefits conversation before slot hold.`,
    };
  }

  // In-network (or language-access path) — find slots and hold
  const slots = await find_slots({
    discipline: analysis.discipline?.[0],
    language: analysis.language === "es" ? "es" : undefined,
    preferences: analysis.preferences ?? undefined,
  });

  // If language-specific search returns nothing, retry without language filter
  const finalSlots =
    slots.data.length === 0 && analysis.language === "es"
      ? (await find_slots({ discipline: analysis.discipline?.[0] })).data
      : slots.data;

  const firstSlot = finalSlots[0];
  let holdId: string | null = null;

  if (firstSlot && !isUnknownInsurance) {
    const hold = await hold_slot({
      slot_id: firstSlot.slot_id,
      patient_ref: patientId ?? `${analysis.child_name ?? "unknown"} (new referral)`,
    });
    holdId = hold.data.hold_id;
  }

  const missingInfo = buildMissingInfo(analysis);
  if (discrepancy) missingInfo.push(discrepancy);

  // Language access: if Spanish-speaking, look up policy and note preferred provider
  let languageNote = "";
  if (analysis.language === "es") {
    await lookup_policy({ topic: "language_access" });
    languageNote = "Family prefers Spanish — match with Spanish-capable provider where possible.";
  }

  const task = await create_task({
    assignee: "intake",
    title: `New ${analysis.discipline?.[0] ?? "therapy"} referral — ${analysis.child_name ?? "unknown child"}`,
    due: tomorrowIso(),
    notes: `${analysis.payer ?? "Insurance unknown"} verified ${insurance.data.status}. ${discrepancy ?? ""} ${languageNote} ${holdId ? `Slot held: ${holdId} (${firstSlot?.start}).` : "No slot held."} ${patientId ? `Existing patient: ${patientId}.` : "New patient."} Parent: ${analysis.parent_name ?? "unknown"}, ${analysis.parent_phone ?? analysis.parent_email ?? "unknown contact"}.`,
  });

  const draftBody = await generateDraft({
    channel: item.channel,
    language: analysis.language,
    recipientName: analysis.parent_name ?? "there",
    childName: analysis.child_name,
    situation: `Referral received for ${analysis.child_name ?? "the child"} for ${analysis.discipline?.[0] ?? "therapy"}. Insurance verified in-network. ${holdId ? `A slot has been held on ${firstSlot?.start} pending staff confirmation.` : "No slot held yet."}${analysis.language === "es" ? " A Spanish-speaking therapist is available." : ""}`,
    constraints: [
      "Do not confirm or schedule any appointment",
      "Tell the parent a staff member will be in touch to confirm a time",
      "Keep it brief and warm",
    ],
    fallback: `Hi ${analysis.parent_name ?? "there"}, thank you for reaching out. We have received the information and a staff member will be in touch shortly to confirm an appointment time.`,
  });

  await draft_message({
    recipient: analysis.parent_email ?? analysis.parent_phone ?? item.sender,
    channel: inferChannel(item),
    body: draftBody,
    language: analysis.language,
  });

  return {
    item_id: item.id,
    classification: patientId ? "existing_patient_request" : "new_referral",
    urgency: "P2",
    requires_human_review: true,
    extracted_intake: buildExtractedIntake(analysis),
    missing_info: missingInfo,
    tools_called: getToolCallsForItem(item.id),
    recommended_next_action: holdId
      ? `Staff should confirm held slot ${holdId} (${firstSlot?.start}) with parent and finalize scheduling.`
      : `Intake should follow up to collect missing info and schedule ${analysis.discipline?.[0] ?? "therapy"} evaluation.`,
    draft_reply: draftBody,
    task_ids: [task.data.task_id],
    escalation: null,
    decision_rationale: `${analysis.payer ?? "Insurance unknown"} verified ${insurance.data.status}. ${discrepancy ?? ""} ${patientId ? "Existing patient found." : "New patient."} ${holdId ? `Slot held (${firstSlot?.start}).` : "No slot held."} ${languageNote} ${missingInfo.length > 0 ? `Missing: ${missingInfo.join(", ")}.` : "All key fields present."}`,
  };
}

async function handleOther(item: InboxItem, analysis: ItemAnalysis): Promise<ItemOutput> {
  const task = await create_task({
    assignee: "front_desk",
    title: `Unclassified inbox item — ${item.subject}`,
    due: tomorrowIso(),
    notes: `Item from ${item.sender} via ${item.channel}. Could not be automatically classified. Requires manual review.`,
  });

  return {
    item_id: item.id,
    classification: "other",
    urgency: "P2",
    requires_human_review: true,
    extracted_intake: buildExtractedIntake(analysis),
    missing_info: ["unable to classify — manual review needed"],
    tools_called: getToolCallsForItem(item.id),
    recommended_next_action: "Front desk should review this item manually.",
    draft_reply: null,
    task_ids: [task.data.task_id],
    escalation: null,
    decision_rationale:
      "Item did not match any known classification pattern. Routed to front desk for manual triage.",
  };
}

// ─── Shared helpers ───────────────────────────────────────────────────────────

function buildExtractedIntake(a: ItemAnalysis) {
  const contact = [a.parent_name, a.parent_phone, a.parent_email]
    .filter(Boolean)
    .join(", ") || null;
  return {
    child_name: a.child_name,
    dob_or_age: a.dob_or_age,
    parent_contact: contact,
    discipline: a.discipline,
    diagnosis_or_concern: a.diagnosis_or_concern,
    payer: a.payer,
    member_id: a.member_id,
  };
}

function buildMissingInfo(a: ItemAnalysis): string[] {
  const missing: string[] = [];
  if (!a.child_name) missing.push("child name");
  if (!a.dob_or_age) missing.push("date of birth");
  if (!a.parent_phone && !a.parent_email) missing.push("parent contact");
  if (!a.payer) missing.push("insurance payer");
  if (!a.member_id) missing.push("member ID");
  return missing;
}

function inferChannel(item: InboxItem): "portal" | "email" | "phone" {
  if (item.channel === "portal_message") return "portal";
  if (item.channel === "email") return "email";
  return "phone";
}

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

function tomorrowIso(): string {
  const d = new Date();
  d.setDate(d.getDate() + 1);
  return d.toISOString().slice(0, 10);
}

function fallbackOutput(item: InboxItem, reason: string): ItemOutput {
  return {
    item_id: item.id,
    classification: "other",
    urgency: "P2",
    requires_human_review: true,
    extracted_intake: {
      child_name: null,
      dob_or_age: null,
      parent_contact: null,
      discipline: null,
      diagnosis_or_concern: null,
      payer: null,
      member_id: null,
    },
    missing_info: [`agent error: ${reason}`],
    tools_called: [],
    recommended_next_action: "Manual review required — agent processing failed for this item.",
    draft_reply: null,
    task_ids: [],
    escalation: null,
    decision_rationale: `Agent threw an unexpected error during processing: ${reason}`,
  };
}