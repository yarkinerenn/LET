import { useState } from "react";
import { Container, Row, Col, Form, Button, Alert } from "react-bootstrap";

const Settings = () => {
    const [openaiApi, setOpenaiApi] = useState(""); // Current OpenAI API Key
    const [grokApi, setGrokApi] = useState("");     // Current Grok API Key
    const [error, setError] = useState("");         // For error messages
    const [success, setSuccess] = useState("");     // For success message

    const handleSubmit = async (e: { preventDefault: () => void }) => {
        e.preventDefault();

        console.log("Submit button clicked!");

        // Ensure at least one API key is filled
        if (!openaiApi && !grokApi) {
            setError("Please enter at least one API key.");
            console.log("Error: Both API fields are empty.");
            return;
        }

        console.log("API keys provided:", { openaiApi, grokApi });

        setError(""); // Clear previous errors
        setSuccess(""); // Clear previous success messages

        // Prepare the request payload (only include non-empty values)
        const payload: { openai_api?: string; grok_api?: string } = {};
        if (openaiApi) payload.openai_api = openaiApi;
        if (grokApi) payload.grok_api = grokApi;

        console.log("Sending request with payload:", payload);

        try {
            const response = await fetch("http://localhost:5000/api/settings/update_api_keys", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                credentials: 'include',
                body: JSON.stringify(payload), // Send only the keys that are filled
            });

            console.log("Response received:", response);

            const result = await response.json();

            if (response.ok) {
                console.log("API keys updated successfully:", result);
                setSuccess(result.message);
                setOpenaiApi(""); // Clear input fields on success
                setGrokApi("");
            } else {
                console.error("Error updating API keys:", result);
                setError(result.error || "An error occurred.");
            }
        } catch (error) {
            console.error("Fetch error:", error);
            setError("Failed to connect to the server.");
        }
    };

    return (
        <Container className="py-5">
            <Row className="justify-content-center">
                <Col md={6} lg={4}>
                    <div className="auth-card">
                        <h2 className="text-center mb-4">Settings</h2>
                        {error && <Alert variant="danger">{error}</Alert>}
                        {success && <Alert variant="success">{success}</Alert>}

                        <Form onSubmit={handleSubmit}>
                            <Form.Group className="mb-3">
                                <Form.Label>OpenAI API Key</Form.Label>
                                <Form.Control
                                    type="text"
                                    placeholder="Enter your OpenAI API key (optional)"
                                    value={openaiApi}
                                    onChange={(e) => setOpenaiApi(e.target.value)}
                                />
                            </Form.Group>

                            <Form.Group className="mb-3">
                                <Form.Label>Grok API Key</Form.Label>
                                <Form.Control
                                    type="text"
                                    placeholder="Enter your Grok API key (optional)"
                                    value={grokApi}
                                    onChange={(e) => setGrokApi(e.target.value)}
                                />
                            </Form.Group>

                            <Button variant="primary" className="w-100 mb-3" type="submit">
                                Update API Keys
                            </Button>
                        </Form>
                    </div>
                </Col>
            </Row>
        </Container>
    );
};

export default Settings;