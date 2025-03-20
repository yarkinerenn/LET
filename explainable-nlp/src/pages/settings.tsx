import { useState } from "react";
import { Container, Row, Col, Form, Button, Alert } from "react-bootstrap";

const Settings = () => {
    const [openaiApi, setOpenaiApi] = useState(""); // Current OpenAI API Key
    const [grokApi, setGrokApi] = useState("");     // Current Grok API Key
    const [error, setError] = useState("");         // For error messages
    const [success, setSuccess] = useState("");     // For success message

    const handleSubmit = async (e: { preventDefault: () => void; }) => {
        e.preventDefault();

        // Make sure both fields are filled
        if (!openaiApi || !grokApi) {
            setError("Please fill in both API keys.");
            return;
        }

        // Send the new API keys to the backend
        const response = await fetch("/api/settings/update_api_keys", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                openai_api: openaiApi,
                grok_api: grokApi,
            }),
        });

        const result = await response.json();

        if (response.ok) {
            setSuccess(result.message);
        } else {
            setError(result.error || "An error occurred.");
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
                                    placeholder="Enter your OpenAI API key"
                                    value={openaiApi}
                                    onChange={(e) => setOpenaiApi(e.target.value)}
                                    required
                                />
                            </Form.Group>

                            <Form.Group className="mb-3">
                                <Form.Label>Grok API Key</Form.Label>
                                <Form.Control
                                    type="text"
                                    placeholder="Enter your Grok API key"
                                    value={grokApi}
                                    onChange={(e) => setGrokApi(e.target.value)}
                                    required
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