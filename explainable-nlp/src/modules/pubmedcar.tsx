// Utility to pretty print PubMed context arrays with labels
export function prettyPubMedContext(
  context: string[] | string,
  labels?: string[]
) {
  // If it's a string, just print as before
  if (typeof context === "string") {
    return <div style={{ whiteSpace: "pre-wrap" }}>{context}</div>;
  }
  // Otherwise, join each context with its label if labels exist
  return (
    <div>
      {context.map((text, i) => (
        <div key={i} style={{ marginBottom: 12 }}>
          {labels?.[i] && (
            <span style={{
              fontWeight: 600,
              color: "#3c4e72",
              display: "block",
              marginBottom: 2
            }}>
              {/* Remove weird chars and spaces */}
              {labels[i].replace(/[\n:]/g, '').trim()}:
            </span>
          )}
          <span style={{ whiteSpace: "pre-wrap" }}>{text.trim()}</span>
        </div>
      ))}
    </div>
  );
}