import express from "express";
import bodyParser from "body-parser";
import { ChatGroq } from "@langchain/groq";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { GROQ_API_KEY } from "./key.js";
import { ROOM_META_TAGS } from "./room_tags.js";
import hotelTagsDef from "./hotel_tags.json" with { type: "json" };

const app = express();
app.use(bodyParser.json());

if (!GROQ_API_KEY) {
  console.error("❌ Missing GROQ_API_KEY in environment");
  process.exit(1);
}

// ✅ Initialize Groq model
const model = new ChatGroq({
  model: "openai/gpt-oss-20b", // or "llama3-70b-8192"
  apiKey: GROQ_API_KEY,
  temperature: 0.7,
  maxTokens: 2000,
});


// ✅ Modern LangChain pipeline
const ROOM_TAG_LABELS = ROOM_META_TAGS.map(([label]) => label);
const ROOM_TAG_OPTIONS = Object.fromEntries(
  ROOM_META_TAGS.filter(([label]) => label !== "Basic Info")
);

const HOTEL_TAG_FIELDS = hotelTagsDef.map(t => t.field);
const HOTEL_TAG_OPTIONS = Object.fromEntries(hotelTagsDef.map(t => [t.field, t.value]));
const HOTEL_YES_NO_FIELDS = hotelTagsDef
  .filter(t => Array.isArray(t.value) && t.value.length === 2 && t.value.includes("yes") && t.value.includes("no"))
  .map(t => t.field);
const HOTEL_NUMERIC_FIELDS = hotelTagsDef
  .filter(t => Array.isArray(t.value) && t.value.every(v => typeof v === "number"))
  .map(t => t.field);
const prompt = ChatPromptTemplate.fromTemplate(`
You are a travel assistant.

Return ONLY a single minified JSON object with exactly these top-level keys: "hotelName", "location", "description", "hotelTags", "rooms".
- No markdown, no code fences, no pre/post text.
- No comments.
- No trailing commas.
- Do not include any extra keys at top level or nested.
- description must be 150-200 words and engaging.

Top-level fields:
- hotelName must equal the input hotel exactly.
- location must equal the input location exactly.

hotelTags schema:
- hotelTags must be an object with exactly these keys:
{hotelTagFields}
- Use only these allowed options for each key (keys correspond to fields):
{hotelTagOptions}
- For keys in {hotelNumericFields}: output a single number from the allowed options.
- For keys in {hotelYesNoFields}: output "yes" or "no".
- For all other keys: output an array of strings, selecting a plausible subset (1-3) from the allowed options. Use [] if not applicable or unknown.

Rooms schema:

- give me all the rooms for this hotel with accurate and exact name:
- Each room item must be an object that includes keys exactly matching these labels:
{roomTagLabels}
- Use these allowed options when applicable (keys correspond to labels):
{roomTagOptions}

Rules for room values:
- For key "Basic Info": the value must be an object {{"Name":"<name>", "Area":"<area (e.g., 350 sq ft)>", "Description":"<short description>"}}.
- For other keys: value must be an array of strings; choose a plausible subset of the allowed options. Use [] if not applicable or unknown.

Input:
hotel: {hotel}
location: {location}
`);

const outputParser = new StringOutputParser();
const chain = prompt.pipe(model).pipe(outputParser);

const descPrompt = ChatPromptTemplate.fromTemplate(`
You are a travel assistant.

Return ONLY a single plain text paragraph describing the hotel "{hotel}" in "{location}".
- No JSON, no markdown, no headings.
- 150-200 words.
- Engaging and informative; mention style, vibe, amenities, location highlights, and ideal guests.
`);
const descChain = descPrompt.pipe(model).pipe(new StringOutputParser());

const roomsPrompt = ChatPromptTemplate.fromTemplate(`
You are a travel assistant.

Return ONLY a single minified JSON array of 2-4 room objects. No top-level key, just the array.
- No markdown, no code fences, no pre/post text.
- Do not include comments or extra fields.
- Never return an empty array. Always include 2-4 rooms.
- Output must be strictly a JSON array, not wrapped in any text.

Each room object must include keys exactly matching these labels:
{roomTagLabels}

Rules:
- For "Basic Info": value must be an object {{"Name":"<name>", "Area":"<area (e.g., 350 sq ft)>", "Description":"<short description>"}}.
- For all other keys: value must be an array of strings selecting a plausible subset from these allowed options:
{roomTagOptions}
- Use [] if a key is not applicable or unknown.

Context:
hotel: {hotel}
location: {location}
`);
const roomsChain = roomsPrompt.pipe(model).pipe(new StringOutputParser());

const hotelTagsPrompt = ChatPromptTemplate.fromTemplate(`
You are a travel assistant.

Return ONLY a single minified JSON object with exactly these keys:
{hotelTagFields}
- Use only these allowed options for each key (keys correspond to fields):
{hotelTagOptions}
- For keys in {hotelNumericFields}: output a single number from the allowed options.
- For keys in {hotelYesNoFields}: output "yes" or "no".
- For all other keys: output an array of strings, selecting a plausible subset (1-3) from the allowed options. Use [] if not applicable or unknown.

Context:
hotel: {hotel}
location: {location}
`);
const hotelTagsChain = hotelTagsPrompt.pipe(model).pipe(new StringOutputParser());

function parseStrictJson(raw) {
  try { return JSON.parse(raw); } catch {}
  const cleaned = String(raw)
    .replace(/```json|```/gi, "")
    .replace(/^[^{]*\{/, "{")
    .replace(/\}[^}]*$/, "}")
    .replace(/[\u201C\u201D]/g, '"')
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/,\s*([}\]])/g, "$1");
  try { return JSON.parse(cleaned); } catch {}
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start !== -1 && end !== -1) {
    try { return JSON.parse(cleaned.slice(start, end + 1)); } catch {}
  }
  return null;
}

function parseArrayJson(raw) {
  try {
    const v = JSON.parse(raw);
    if (Array.isArray(v)) return v;
    if (v && typeof v === "object" && Array.isArray(v.rooms)) return v.rooms;
  } catch {}
  const cleaned = String(raw)
    .replace(/```json|```/gi, "")
    .replace(/^[^\[]*\[/, "[")
    .replace(/\][^\]]*$/, "]");
  try {
    const v = JSON.parse(cleaned);
    if (Array.isArray(v)) return v;
    if (v && typeof v === "object" && Array.isArray(v.rooms)) return v.rooms;
  } catch {}
  // Try to regex-extract an array literal
  const match = cleaned.match(/\[[\s\S]*\]/);
  if (match) {
    try {
      const v = JSON.parse(match[0]);
      if (Array.isArray(v)) return v;
    } catch {}
  }
  return null;
}

// console.log("chain", chain);
// ✅ API Endpoint
app.post("/hotel-info", async (req, res) => {
    // console.log("req.body", req.body);
  try {
    const { hotelName, location } = req.body;

    if (!hotelName || !location) {
      return res.status(400).json({ error: "hotelName and location are required" });
    }

    console.log("hotelName", hotelName);
    console.log("location", location);

    const rawResponse = await chain.invoke({
      hotel: hotelName,
      location,
      roomTagLabels: JSON.stringify(ROOM_TAG_LABELS),
      roomTagOptions: JSON.stringify(ROOM_TAG_OPTIONS),
      hotelTagFields: JSON.stringify(HOTEL_TAG_FIELDS),
      hotelTagOptions: JSON.stringify(HOTEL_TAG_OPTIONS),
      hotelYesNoFields: JSON.stringify(HOTEL_YES_NO_FIELDS),
      hotelNumericFields: JSON.stringify(HOTEL_NUMERIC_FIELDS),
    });

    // console.log("rawResponse", rawResponse);

    const parsed = parseStrictJson(rawResponse);
    let result = parsed && typeof parsed === "object"
      ? parsed
      : {
          hotelName,
          location,
          description: "",
          hotelTags: {},
          rooms: []
        };

    // Normalize mandatory fields
    result.hotelName = hotelName;
    result.location = location;
    if (!result.hotelTags || typeof result.hotelTags !== "object") result.hotelTags = {};
    if (!Array.isArray(result.rooms)) result.rooms = [];

    // Ensure description is always present and AI-generated
    if (!result.description || typeof result.description !== "string" || result.description.trim().length < 50) {
      try {
        const genDesc = await descChain.invoke({ hotel: hotelName, location });
        result.description = String(genDesc).trim();
      } catch (_) {
        // Fallback to raw model output if description generation fails
        result.description = typeof rawResponse === "string" ? rawResponse.trim() : "";
      }
    }

    // Ensure rooms are present and follow schema using AI if needed
    if (!Array.isArray(result.rooms) || result.rooms.length < 2) {
      try {
        const roomsRaw = await roomsChain.invoke({
          hotel: hotelName,
          location,
          roomTagLabels: JSON.stringify(ROOM_TAG_LABELS),
          roomTagOptions: JSON.stringify(ROOM_TAG_OPTIONS),
        });
        const roomsParsed = parseArrayJson(roomsRaw);
        if (Array.isArray(roomsParsed)) {
          result.rooms = roomsParsed;
        }
      } catch (_) {
        // ignore, will normalize whatever is present
      }
    }

    // Normalize each room to include all required keys and shapes
    if (Array.isArray(result.rooms)) {
      const requiredLabels = ROOM_TAG_LABELS;
      result.rooms = result.rooms.slice(0, 4).map((room, idx) => {
        const normalized = {};
        for (const label of requiredLabels) {
          if (label === "Basic Info") {
            const bi = room && room["Basic Info"] ? room["Basic Info"] : {};
            normalized["Basic Info"] = {
              Name: bi && typeof bi.Name === "string" ? bi.Name : `Room ${idx + 1}`,
              Area: bi && typeof bi.Area === "string" ? bi.Area : "",
              Description: bi && typeof bi.Description === "string" ? bi.Description : ""
            };
          } else {
            let v = room ? room[label] : undefined;
            if (typeof v === "string") v = [v];
            if (!Array.isArray(v)) v = [];
            normalized[label] = v.filter(x => typeof x === "string");
          }
        }
        return normalized;
      });
      if (result.rooms.length === 0) {
        // Create two skeleton rooms to avoid empty array
        const makeSkeleton = (index) => {
          const o = {};
          for (const label of requiredLabels) {
            if (label === "Basic Info") {
              o["Basic Info"] = { Name: `Room ${index + 1}`, Area: "", Description: "" };
            } else {
              o[label] = [];
            }
          }
          return o;
        };
        result.rooms = [makeSkeleton(0), makeSkeleton(1)];
      } else if (result.rooms.length < 2) {
        // If still less than 2, duplicate first to meet min length
        while (result.rooms.length < 2 && result.rooms.length > 0) {
          result.rooms.push({ ...result.rooms[0] });
        }
      }
    } else {
      result.rooms = [];
    }

    // Ensure hotelTags are present using AI if missing or empty
    const hotelTagFieldsArr = HOTEL_TAG_FIELDS;
    const isHotelTagsEmpty =
      !result.hotelTags ||
      typeof result.hotelTags !== "object" ||
      Object.keys(result.hotelTags).length === 0 ||
      hotelTagFieldsArr.every(k => !(k in result.hotelTags));

    if (isHotelTagsEmpty) {
      try {
        const hotelTagsRaw = await hotelTagsChain.invoke({
          hotel: hotelName,
          location,
          hotelTagFields: JSON.stringify(HOTEL_TAG_FIELDS),
          hotelTagOptions: JSON.stringify(HOTEL_TAG_OPTIONS),
          hotelYesNoFields: JSON.stringify(HOTEL_YES_NO_FIELDS),
          hotelNumericFields: JSON.stringify(HOTEL_NUMERIC_FIELDS),
        });
        const hotelTagsParsed = parseStrictJson(hotelTagsRaw);
        if (hotelTagsParsed && typeof hotelTagsParsed === "object") {
          result.hotelTags = hotelTagsParsed;
        }
      } catch (_) {
        // ignore
      }
    }

    // Normalize hotelTags to match schema and allowed types/options
    if (!result.hotelTags || typeof result.hotelTags !== "object") result.hotelTags = {};
    const normalizedHotelTags = {};
    for (const field of HOTEL_TAG_FIELDS) {
      const allowed = HOTEL_TAG_OPTIONS[field];
      const isYesNo = HOTEL_YES_NO_FIELDS.includes(field);
      const isNumeric = HOTEL_NUMERIC_FIELDS.includes(field);
      let value = result.hotelTags[field];
      if (isNumeric) {
        if (typeof value !== "number" || !Array.isArray(allowed) || !allowed.includes(value)) {
          const nums = Array.isArray(allowed) ? allowed : [];
          const chosen = typeof value === "number" ? (nums.find(n => n === value) ?? nums[0]) : nums[0];
          value = typeof chosen === "number" ? chosen : (nums[0] ?? 0);
        }
        normalizedHotelTags[field] = value;
      } else if (isYesNo) {
        const v = String(value || "").toLowerCase();
        normalizedHotelTags[field] = v === "yes" ? "yes" : v === "no" ? "no" : "no";
      } else {
        let arr = Array.isArray(value) ? value : (typeof value === "string" ? [value] : []);
        arr = arr.filter(v => typeof v === "string" && Array.isArray(allowed) && allowed.includes(v));
        normalizedHotelTags[field] = arr.slice(0, 3);
      }
    }
    result.hotelTags = normalizedHotelTags;

    res.json(result);
  } catch (err) {
    console.error("❌ Error:", err.message);
    res.status(500).json({ error: "Something went wrong" });
  }
});

// ✅ Start Server
app.listen(8000, () => {
  console.log("🚀 Server running at http://localhost:8000");
});
