import fetch from "node-fetch";

const INSTANCE_ID = "ins_34Qzpt3QAvRKYI31kMsUUHzyefg";
const SECRET = process.env.CLERK_SECRET_KEY;
const API_BASE = process.env.CLERK_API_BASE || "https://working-kangaroo-26.clerk.accounts.dev";

if (!SECRET) {
  console.error("CLERK_SECRET_KEY is not set in this shell.");
  process.exit(1);
}

const payload = {
  allowed_origins: [
    "http://127.0.0.1:5510",
    "http://localhost:5510",
  ],
};

async function main() {
  const url = `${API_BASE.replace(/\/$/, "")}/v1/instances/${INSTANCE_ID}`;

  const res = await fetch(url, {
    method: "PATCH",
    headers: {
      Authorization: `Bearer ${SECRET}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const text = await res.text();

  if (!res.ok) {
    console.error("Failed to update Clerk instance:", res.status, res.statusText);
    console.error(text);
    process.exit(1);
  }

  console.log("Clerk instance updated:");
  console.log(text);
}

main().catch(err => {
  console.error("Unexpected error while updating Clerk instance:", err);
  process.exit(1);
});
