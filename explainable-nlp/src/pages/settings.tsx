import React, { useState } from "react";
import {Container, Row, Col, Form, Button, Alert, ButtonGroup, ToggleButton} from "react-bootstrap";
import {useProvider} from "../modules/provider";

const Settings = () => {
    const [openaiApi, setOpenaiApi] = useState(""); // Current OpenAI API Key
    const [grokApi, setGrokApi] = useState("");     // Current Grok API Key
    const [error, setError] = useState("");         // For error messages
    const [success, setSuccess] = useState("");     // For success message
    const groqModels = [
        { name: "allam-2-7b" },
        { name: "deepseek-r1-distill-llama-70b" },
        { name: "deepseek-r1-distill-qwen-32b" },
        { name: "gemma2-9b-it" },
        { name: "llama-3.1-8b-instant" },
        { name: "llama-3.2-11b-vision-preview" },
        { name: "llama-3.2-1b-preview" },
        { name: "llama-3.2-3b-preview" },
        { name: "llama-3.2-90b-vision-preview" },
        { name: "llama-3.3-70b-specdec" },
        { name: "llama-3.3-70b-versatile" },
        { name: "llama-guard-3-8b" },
        { name: "llama3-70b-8192" },
        { name: "llama3-8b-8192" },
        { name: "mistral-saba-24b" },
        { name: "qwen-2.5-32b" },
        { name: "qwen-2.5-coder-32b" },
        { name: "qwen-qwq-32b" }
    ];
    const { provider, setProvider, model, setModel } = useProvider();

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

                            <Button variant="dark" className="w-100 mb-3" type="submit">
                                Update API Keys
                            </Button>
                        </Form>
                    </div>
                </Col>
            </Row>
            <div className="mt-4 p-4 rounded bg-light">
                <h5 className="mb-3">AI Settings</h5>
                <p className="text-muted mb-3">Select the AI provider for classification and explanation:</p>

                <ButtonGroup className="d-flex justify-content-start">
                    <ToggleButton
                        id="provider-openai"
                        type="radio"
                        variant={provider === 'openai' ? 'dark' : 'outline-primary'}
                        name="provider"
                        value="openai"
                        checked={provider === 'openai'}
                        onChange={(e) => setProvider(e.currentTarget.value)}
                        className="me-3 mb-2"
                    >
                        OpenAI
                    </ToggleButton>
                    <ToggleButton
                        id="provider-groq"
                        type="radio"
                        variant={provider === 'groq' ? 'dark' : 'outline-primary'}
                        name="provider"
                        value="groq"
                        checked={provider === 'groq'}
                        onChange={(e) => setProvider(e.currentTarget.value)}
                        className="mb-2"
                    >
                        Groq
                    </ToggleButton>
                </ButtonGroup>
                {/* Show model selection only if Groq is chosen */}
                {(provider === 'groq') && (
                    <div className="mb-3">
                        <span className="me-3">Select Model:</span>
                        <Form.Select value={model}
                                     onChange={(e) => setModel(e.target.value)}>
                            <option value="">-- Select a Model --</option>
                            {groqModels.map((m) => (
                                <option key={m.name} value={m.name}>
                                    {m.name}
                                </option>
                            ))}
                        </Form.Select>
                    </div>
                )}
            </div>
        </Container>
    );
};

export default Settings;