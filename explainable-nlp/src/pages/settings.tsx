import React, { useState } from "react";
import {Container, Row, Col, Form, Button, Alert, ButtonGroup, ToggleButton} from "react-bootstrap";
import {useProvider} from "../modules/provider";

const Settings = () => {
    const [openaiApi, setOpenaiApi] = useState(""); // Current OpenAI API Key
    const [grokApi, setGrokApi] = useState("");     // Current Grok API Key
    const [deepseekApi, setdeepseekApi] = useState("");
    const [openrouterApi, setopenrouterApi] = useState("");
    const [error, setError] = useState("");         // For error messages
    const [success, setSuccess] = useState("");     // For success message
    const openAIModels=[
        { name: "gpt-4.1-2025-04-14" },
        { name: "o4-mini-2025-04-16" },
        { name: "gpt-4.1-nano-2025-04-14" },
        {name:"gpt-3.5-turbo"}
    ];
    // Ollama models
    const ollamaModels = [
        { name: "jsk/bio-mistral" }
    ];
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
    const openrouterModels = [
        { name: "deepseek/deepseek-r1-0528-qwen3-8b:free" },
        { name: "deepseek-r1-0528" },
        { name: "sarvam-m" },
        { name: "devstral-small" },
        { name: "gemma-3n-e4b-it" },
        { name: "llama-3.3-8b-instruct" },
        { name: "deephermes-3-mistral-24b-preview" },
        { name: "phi-4-reasoning-plus" },
        { name: "phi-4-reasoning" },
        { name: "internvl3-14b" },
        { name: "internvl3-2b" },
        { name: "deepseek-prover-v2" },
        { name: "qwen3-30b-a3b" },
        { name: "qwen3-8b" },
        { name: "qwen3-14b" },
        { name: "qwen3-32b" },
        { name: "qwen3-235b-a22b" },
        { name: "deepseek-r1t-chimera" },
        { name: "mai-ds-r1" },
        { name: "glm-z1-32b" },
        { name: "glm-4-32b" },
        { name: "shisa-v2-llama3.3-70b" },
        { name: "qwq-32b-arliai-rpr-v1" },
        { name: "deepcoder-14b-preview" },
        { name: "kimi-vl-a3b-thinking" },
        { name: "llama-3.3-nemotron-super-49b-v1" },
        { name: "llama-3.1-nemotron-ultra-253b-v1" },
        { name: "llama-4-maverick" },
        { name: "llama-4-scout" },
        { name: "deepseek-v3-base" },
        { name: "qwen2.5-vl-3b-instruct" },
        { name: "gemini-2.5-pro-exp-03-25" },
        { name: "qwen2.5-vl-32b-instruct" },
        { name: "deepseek-chat-v3-0324" },
        { name: "qwerky-72b" },
        { name: "mistral-small-3.1-24b-instruct" },
        { name: "olympiccoder-32b" },
        { name: "gemma-3-1b-it" },
        { name: "gemma-3-4b-it" },
        { name: "gemma-3-12b-it" },
        { name: "reka-flash-3" },
        { name: "gemma-3-27b-it" },
        { name: "deepseek-r1-zero" },
        { name: "qwq-32b" },
        { name: "moonlight-16b-a3b-instruct" },
        { name: "deephermes-3-llama-3-8b-preview" },
        { name: "dolphin3.0-r1-mistral-24b" },
        { name: "dolphin3.0-mistral-24b" },
        { name: "qwen2.5-vl-72b-instruct" },
        { name: "mistral-small-24b-instruct-2501" },
        { name: "deepseek-r1-distill-qwen-32b" },
        { name: "deepseek-r1-distill-qwen-14b" },
        { name: "deepseek-r1-distill-llama-70b" },
        { name: "deepseek-r1" },
        { name: "deepseek-chat" },
        { name: "gemini-2.0-flash-exp" },
        { name: "llama-3.3-70b-instruct" },
        { name: "qwen-2.5-coder-32b-instruct" },
        { name: "qwen-2.5-7b-instruct" },
        { name: "llama-3.2-3b-instruct" },
        { name: "llama-3.2-1b-instruct" },
        { name: "llama-3.2-11b-vision-instruct" },
        { name: "qwen-2.5-72b-instruct" },
        { name: "qwen-2.5-vl-7b-instruct" },
        { name: "llama-3.1-405b" },
        { name: "llama-3.1-8b-instruct" },
        { name: "mistral-nemo" },
        { name: "gemma-2-9b-it" },
        { name: "mistral-7b-instruct" }
    ];
    const{ provider, setProvider,providerex, setProviderex, model, setModel, modelex, setModelex } = useProvider();
    const handleExplanationSettingsUpdate = async () => {
        const payload = {
            preferred_providerex: providerex,
            preferred_modelex: modelex
        };

        try {
            const response = await fetch("http://localhost:5000/api/settings/update_preferred_explanation", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                credentials: 'include',
                body: JSON.stringify(payload),
            });

            const result = await response.json();

            if (response.ok) {
                setSuccess("Explanation preferences updated successfully!");
            } else {
                setError(result.error || "An error occurred while updating explanation settings.");
            }
        } catch (error) {
            console.error("Explanation update error:", error);
            setError("Failed to connect to the server.");
        }
    };
    const handleClassificationSettingsUpdate = async () => {
        const payload = {
            preferred_provider: provider,
            preferred_model: model
        };

        try {
            const response = await fetch("http://localhost:5000/api/settings/update_preferred_classification", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                credentials: 'include',
                body: JSON.stringify(payload),
            });

            const result = await response.json();

            if (response.ok) {
                setSuccess("Classification preferences updated successfully!");
            } else {
                setError(result.error || "An error occurred while updating classification settings.");
            }
        } catch (error) {
            console.error("Classification update error:", error);
            setError("Failed to connect to the server.");
        }
    };
    const handleSubmit = async (e: { preventDefault: () => void }) => {
        e.preventDefault();

        console.log("Submit button clicked!");

        // Ensure at least one API key is filled
        if (!openaiApi && !grokApi && !deepseekApi && !openrouterApi) {
            setError("Please enter at least one API key.");
            console.log("Error: Both API fields are empty.");
            return;
        }

        console.log("API keys provided:", { openaiApi, grokApi });

        setError(""); // Clear previous errors
        setSuccess(""); // Clear previous success messages

        // Prepare the request payload (only include non-empty values)
        const payload: { openai_api?: string; grok_api?: string ,deepseek_api?: string, openrouter_api?: string} = {};
        if (openaiApi) payload.openai_api = openaiApi;
        if (grokApi) payload.grok_api = grokApi;
        if (deepseekApi) payload.deepseek_api = deepseekApi;
        if (openrouterApi) payload.openrouter_api = openrouterApi;


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

                            <Form.Group className="mb-3">
                                <Form.Label>DeepSeek API Key</Form.Label>
                                <Form.Control
                                    type="text"
                                    placeholder="Enter your Deepseek API key (optional)"
                                    value={deepseekApi}
                                    onChange={(e) => setdeepseekApi(e.target.value)}
                                />
                            </Form.Group>

                            <Form.Group className="mb-3">
                                <Form.Label>Openrouter API Key</Form.Label>
                                <Form.Control
                                    type="text"
                                    placeholder="Enter your Openrouter API key (optional)"
                                    value={openrouterApi}
                                    onChange={(e) => setopenrouterApi(e.target.value)}
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
                <h5 className="mb-3">Classification Settings</h5>
                <p className="text-muted mb-3">Select the AI provider for classification</p>

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
                        className="mb-2 me-3"
                    >
                        Groq
                    </ToggleButton>
                    <ToggleButton
                        id="provider-deepseek"
                        type="radio"
                        variant={provider === 'deepseek' ? 'dark' : 'outline-primary'}
                        name="provider"
                        value="deepseek"
                        checked={provider === 'deepseek'}
                        onChange={(e) => setProvider(e.currentTarget.value)}
                        className="mb-2 me-3"
                    >
                        Deepseek
                    </ToggleButton>
                    <ToggleButton
                        id="provider-openrouter"
                        type="radio"
                        variant={provider === 'openrouter' ? 'dark' : 'outline-primary'}
                        name="provider"
                        value="openrouter"
                        checked={provider === 'openrouter'}
                        onChange={(e) => setProvider(e.currentTarget.value)}
                        className="mb-2 me-3"
                    >
                        Open router
                    </ToggleButton>
                    <ToggleButton
                        id="provider-ollama"
                        type="radio"
                        variant={provider === 'ollama' ? 'dark' : 'outline-primary'}
                        name="provider"
                        value="ollama"
                        checked={provider === 'ollama'}
                        onChange={(e) => setProvider(e.currentTarget.value)}
                        className="mb-2"
                    >
                        Ollama
                    </ToggleButton>
                </ButtonGroup>

                {provider === 'groq' && (
                    <div className="mb-3">
                        <span className="me-3">Select Model:</span>
                        <Form.Select value={model} onChange={(e) => setModel(e.target.value)}>
                            <option value="">-- Select a Model --</option>
                            {groqModels.map((m) => (
                                <option key={m.name} value={m.name}>
                                    {m.name}
                                </option>
                            ))}
                        </Form.Select>
                    </div>
                )}
                {provider === 'openai' && (
                    <div className="mb-3">
                        <span className="me-3">Select Model:</span>
                        <Form.Select value={model} onChange={(e) => setModel(e.target.value)}>
                            <option value="">-- Select a Model --</option>
                            {openAIModels.map((m) => (
                                <option key={m.name} value={m.name}>
                                    {m.name}
                                </option>
                            ))}
                        </Form.Select>
                    </div>
                )}
                {provider === 'openrouter' && (
                    <div className="mb-3">
                        <span className="me-3">Select Model:</span>
                        <Form.Select value={model} onChange={(e) => setModel(e.target.value)}>
                            <option value="">-- Select a Model --</option>
                            {openrouterModels.map((m) => (
                                <option key={m.name} value={m.name}>
                                    {m.name}
                                </option>
                            ))}
                        </Form.Select>
                    </div>
                )}
                {provider === 'ollama' && (
                    <div className="mb-3">
                        <span className="me-3">Select Model:</span>
                        <Form.Select value={model} onChange={(e) => setModel(e.target.value)}>
                            <option value="">-- Select a Model --</option>
                            {ollamaModels.map((m) => (
                                <option key={m.name} value={m.name}>
                                    {m.name}
                                </option>
                            ))}
                        </Form.Select>
                    </div>
                )}

                <Button
                    variant="dark"
                    className="mt-3"
                    onClick={handleClassificationSettingsUpdate}
                >
                    Save Classification Preferences
                </Button>
            </div>
            <div className="mt-4 p-4 rounded bg-light">
                <h5 className="mb-3">Explanation Settings</h5>
                <p className="text-muted mb-3">Select the AI provider for explanation</p>

                <ButtonGroup className="d-flex justify-content-start">
                    <ToggleButton
                        id="providerex-openai"
                        type="radio"
                        variant={providerex === 'openai' ? 'dark' : 'outline-primary'}
                        name="providerex"
                        value="openai"
                        checked={providerex === 'openai'}
                        onChange={(e) => setProviderex(e.currentTarget.value)}
                        className="me-3 mb-2"
                    >
                        OpenAI
                    </ToggleButton>
                    <ToggleButton
                        id="providerex-groq"
                        type="radio"
                        variant={providerex === 'groq' ? 'dark' : 'outline-primary'}
                        name="providerex"
                        value="groq"
                        checked={providerex === 'groq'}
                        onChange={(e) => setProviderex(e.currentTarget.value)}
                        className="mb-2 me-3"
                    >
                        Groq
                    </ToggleButton>
                    <ToggleButton
                        id="providerex-deepseek"
                        type="radio"
                        variant={providerex === 'deepseek' ? 'dark' : 'outline-primary'}
                        name="providerex"
                        value="deepseek"
                        checked={providerex === 'deepseek'}
                        onChange={(e) => setProviderex(e.currentTarget.value)}
                        className="mb-2 me-3"
                    >
                        Deepseek
                    </ToggleButton>
                    <ToggleButton
                        id="providerex-openrouter"
                        type="radio"
                        variant={providerex === 'openrouter' ? 'dark' : 'outline-primary'}
                        name="providerex"
                        value="openrouter"
                        checked={providerex === 'openrouter'}
                        onChange={(e) => setProviderex(e.currentTarget.value)}
                        className="mb-2 me-3"
                    >
                        Openrouter
                    </ToggleButton>
                    <ToggleButton
                        id="providerex-ollama"
                        type="radio"
                        variant={providerex === 'ollama' ? 'dark' : 'outline-primary'}
                        name="providerex"
                        value="ollama"
                        checked={providerex === 'ollama'}
                        onChange={(e) => setProviderex(e.currentTarget.value)}
                        className="mb-2"
                    >
                        Ollama
                    </ToggleButton>
                </ButtonGroup>

                {providerex === 'groq' && (
                    <div className="mb-3">
                        <span className="me-3">Select Model:</span>
                        <Form.Select value={modelex} onChange={(e) => setModelex(e.target.value)}>
                            <option value="">-- Select a Model --</option>
                            {groqModels.map((m) => (
                                <option key={m.name} value={m.name}>
                                    {m.name}
                                </option>
                            ))}
                        </Form.Select>
                    </div>
                )}
                {providerex === 'openai' && (
                    <div className="mb-3">
                        <span className="me-3">Select Model:</span>
                        <Form.Select value={modelex} onChange={(e) => setModelex(e.target.value)}>
                            <option value="">-- Select a Model --</option>
                            {openAIModels.map((m) => (
                                <option key={m.name} value={m.name}>
                                    {m.name}
                                </option>
                            ))}
                        </Form.Select>
                    </div>
                )}
                {providerex === 'openrouter' && (
                    <div className="mb-3">
                        <span className="me-3">Select Model:</span>
                        <Form.Select value={modelex} onChange={(e) => setModelex(e.target.value)}>
                            <option value="">-- Select a Model --</option>
                            {openrouterModels.map((m) => (
                                <option key={m.name} value={m.name}>
                                    {m.name}
                                </option>
                            ))}
                        </Form.Select>
                    </div>
                )}
                {providerex === 'ollama' && (
                    <div className="mb-3">
                        <span className="me-3">Select Model:</span>
                        <Form.Select value={modelex} onChange={(e) => setModelex(e.target.value)}>
                            <option value="">-- Select a Model --</option>
                            {ollamaModels.map((m) => (
                                <option key={m.name} value={m.name}>
                                    {m.name}
                                </option>
                            ))}
                        </Form.Select>
                    </div>
                )}

                <Button
                    variant="dark"
                    className="mt-3"
                    onClick={handleExplanationSettingsUpdate}
                >
                    Save Explanation Preferences
                </Button>
            </div>
        </Container>
    );
};

export default Settings;