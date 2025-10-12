import React, { useState } from "react";
import {Container, Row, Col, Form, Button, Alert, ButtonGroup, ToggleButton} from "react-bootstrap";
import {useProvider} from "../modules/provider";

const Settings = () => {
    const [openaiApi, setOpenaiApi] = useState(""); // Current OpenAI API Key
    const [grokApi, setGrokApi] = useState("");     // Current Grok API Key
    const [deepseekApi, setdeepseekApi] = useState("");
    const [openrouterApi, setopenrouterApi] = useState("");
    const [geminiApi, setGeminiApi] = useState("");
    const [error, setError] = useState("");         // For error messages
    const [success, setSuccess] = useState("");     // For success message
    const openAIModels=[
        { name: "gpt-5-2025-08-07" },
        { name: "o4-mini-2025-04-16" },
        { name: "gpt-4.1-nano-2025-04-14" },
        {name:"gpt-3.5-turbo"},
        {name: "gpt-4o-mini-2024-07-18"},
        {name:"gpt-5-nano-2025-08-07"},
        {name:"gpt-5-mini-2025-08-07"}

    ];
    // Ollama models
    const ollamaModels = [
        { name: "jsk/bio-mistral" },
        {name:"phi3.5:latest"},
        {name:"gemma:2b"},
        {name:"llama3.1:8b"},
        {name:"mistral:7b"},
    ];
    const groqModels = [
        { name: "allam-2-7b" },
        { name: "llama-3.3-70b-versatile" },
        { name: "llama-3.1-8b-instant" },

    ];
    const openrouterModels = [
        { name: "deepseek/deepseek-r1-0528-qwen3-8b:free" },
        { name: "deepseek-r1-0528" },
        { name: "sarvam-m" },
        { name: "devstral-small" },
        { name: "gemma-3n-e4b-it" },
        { name: "google/gemma-3n-e2b-it:free" },
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

    const geminiModels = [
        { name: "models/gemini-1.5-flash-8b" },
        { name: "gemini-1.5-flash" },
        { name: "gemini-2.0-flash-exp" },
        { name: "gemini-2.5-pro-exp-03-25" }
    ];
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
        if (!openaiApi && !grokApi && !deepseekApi && !openrouterApi && !geminiApi) {
            setError("Please enter at least one API key.");
            console.log("Error: Both API fields are empty.");
            return;
        }

        console.log("API keys provided:", { openaiApi, grokApi });

        setError(""); // Clear previous errors
        setSuccess(""); // Clear previous success messages

        // Prepare the request payload (only include non-empty values)
        const payload: { openai_api?: string; grok_api?: string; deepseek_api?: string; openrouter_api?: string; gemini_api?: string } = {};
        if (openaiApi) payload.openai_api = openaiApi;
        if (grokApi) payload.grok_api = grokApi;
        if (deepseekApi) payload.deepseek_api = deepseekApi;
        if (openrouterApi) payload.openrouter_api = openrouterApi;
        if (geminiApi) payload.gemini_api = geminiApi;


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
                setGeminiApi("");
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
            <h2 className="text-center mb-5">Settings</h2>
            
            {/* Global alerts */}
            {error && <Alert variant="danger" className="mb-4">{error}</Alert>}
            {success && <Alert variant="success" className="mb-4">{success}</Alert>}

            <Row className="g-4">
                {/* Left Column - API Keys */}
                <Col lg={4}>
                    <div className="h-100 p-4 rounded bg-white border shadow-sm">
                        <h5 className="mb-4 text-primary">
                            <i className="fas fa-key me-2"></i>
                            API Keys
                        </h5>
                        <p className="text-muted mb-4">Configure your API keys for different providers</p>

                        <Form onSubmit={handleSubmit}>
                            <Form.Group className="mb-3">
                                <Form.Label className="fw-semibold">OpenAI API Key</Form.Label>
                                <Form.Control
                                    type="password"
                                    placeholder="Enter your OpenAI API key"
                                    value={openaiApi}
                                    onChange={(e) => setOpenaiApi(e.target.value)}
                                    className="border-0 bg-light"
                                />
                            </Form.Group>

                            <Form.Group className="mb-3">
                                <Form.Label className="fw-semibold">Grok API Key</Form.Label>
                                <Form.Control
                                    type="password"
                                    placeholder="Enter your Grok API key"
                                    value={grokApi}
                                    onChange={(e) => setGrokApi(e.target.value)}
                                    className="border-0 bg-light"
                                />
                            </Form.Group>

                            <Form.Group className="mb-3">
                                <Form.Label className="fw-semibold">DeepSeek API Key</Form.Label>
                                <Form.Control
                                    type="password"
                                    placeholder="Enter your Deepseek API key"
                                    value={deepseekApi}
                                    onChange={(e) => setdeepseekApi(e.target.value)}
                                    className="border-0 bg-light"
                                />
                            </Form.Group>

                            <Form.Group className="mb-3">
                                <Form.Label className="fw-semibold">Openrouter API Key</Form.Label>
                                <Form.Control
                                    type="password"
                                    placeholder="Enter your Openrouter API key"
                                    value={openrouterApi}
                                    onChange={(e) => setopenrouterApi(e.target.value)}
                                    className="border-0 bg-light"
                                />
                            </Form.Group>

                            <Form.Group className="mb-4">
                                <Form.Label className="fw-semibold">Gemini API Key</Form.Label>
                                <Form.Control
                                    type="password"
                                    placeholder="Enter your Gemini API key"
                                    value={geminiApi}
                                    onChange={(e) => setGeminiApi(e.target.value)}
                                    className="border-0 bg-light"
                                />
                            </Form.Group>

                            <Button 
                                variant="primary" 
                                className="w-100 py-2" 
                                type="submit"
                                size="lg"
                            >
                                <i className="fas fa-save me-2"></i>
                                Update API Keys
                            </Button>
                        </Form>
                    </div>
                </Col>

                {/* Right Column - Provider Settings */}
                <Col lg={8}>
                    <Row className="g-4">
                        {/* Classification Settings */}
                        <Col xs={12}>
                            <div className="h-100 p-4 rounded bg-white border shadow-sm">
                                <h5 className="mb-4 text-success">
                                    <i className="fas fa-brain me-2"></i>
                                    Classification Settings
                                </h5>
                                <p className="text-muted mb-4">Select the AI provider for classification tasks</p>

                                <div className="mb-4">
                                    <h6 className="mb-3">Provider Selection</h6>
                                    <ButtonGroup className="d-flex flex-wrap">
                                        <ToggleButton
                                            id="provider-openai"
                                            type="radio"
                                            variant={provider === 'openai' ? 'primary' : 'outline-primary'}
                                            name="provider"
                                            value="openai"
                                            checked={provider === 'openai'}
                                            onChange={(e) => setProvider(e.currentTarget.value)}
                                            className="me-2 mb-2"
                                        >
                                            OpenAI
                                        </ToggleButton>
                                        <ToggleButton
                                            id="provider-groq"
                                            type="radio"
                                            variant={provider === 'groq' ? 'primary' : 'outline-primary'}
                                            name="provider"
                                            value="groq"
                                            checked={provider === 'groq'}
                                            onChange={(e) => setProvider(e.currentTarget.value)}
                                            className="me-2 mb-2"
                                        >
                                            Groq
                                        </ToggleButton>
                                        <ToggleButton
                                            id="provider-deepseek"
                                            type="radio"
                                            variant={provider === 'deepseek' ? 'primary' : 'outline-primary'}
                                            name="provider"
                                            value="deepseek"
                                            checked={provider === 'deepseek'}
                                            onChange={(e) => setProvider(e.currentTarget.value)}
                                            className="me-2 mb-2"
                                        >
                                            Deepseek
                                        </ToggleButton>
                                        <ToggleButton
                                            id="provider-openrouter"
                                            type="radio"
                                            variant={provider === 'openrouter' ? 'primary' : 'outline-primary'}
                                            name="provider"
                                            value="openrouter"
                                            checked={provider === 'openrouter'}
                                            onChange={(e) => setProvider(e.currentTarget.value)}
                                            className="me-2 mb-2"
                                        >
                                            Openrouter
                                        </ToggleButton>
                                        <ToggleButton
                                            id="provider-gemini"
                                            type="radio"
                                            variant={provider === 'gemini' ? 'primary' : 'outline-primary'}
                                            name="provider"
                                            value="gemini"
                                            checked={provider === 'gemini'}
                                            onChange={(e) => setProvider(e.currentTarget.value)}
                                            className="me-2 mb-2"
                                        >
                                            Gemini
                                        </ToggleButton>
                                        <ToggleButton
                                            id="provider-ollama"
                                            type="radio"
                                            variant={provider === 'ollama' ? 'primary' : 'outline-primary'}
                                            name="provider"
                                            value="ollama"
                                            checked={provider === 'ollama'}
                                            onChange={(e) => setProvider(e.currentTarget.value)}
                                            className="mb-2"
                                        >
                                            Ollama
                                        </ToggleButton>
                                    </ButtonGroup>
                                </div>

                                {/* Model Selection */}
                                {(provider === 'gemini' || provider === 'groq' || provider === 'openai' || provider === 'openrouter' || provider === 'ollama') && (
                                    <div className="mb-4">
                                        <h6 className="mb-3">Model Selection</h6>
                                        <Form.Select 
                                            value={model} 
                                            onChange={(e) => setModel(e.target.value)}
                                            className="border-0 bg-light"
                                        >
                                            <option value="">-- Select a Model --</option>
                                            {provider === 'gemini' && geminiModels.map((m) => (
                                                <option key={m.name} value={m.name}>
                                                    {m.name}
                                                </option>
                                            ))}
                                            {provider === 'groq' && groqModels.map((m) => (
                                                <option key={m.name} value={m.name}>
                                                    {m.name}
                                                </option>
                                            ))}
                                            {provider === 'openai' && openAIModels.map((m) => (
                                                <option key={m.name} value={m.name}>
                                                    {m.name}
                                                </option>
                                            ))}
                                            {provider === 'openrouter' && openrouterModels.map((m) => (
                                                <option key={m.name} value={m.name}>
                                                    {m.name}
                                                </option>
                                            ))}
                                            {provider === 'ollama' && ollamaModels.map((m) => (
                                                <option key={m.name} value={m.name}>
                                                    {m.name}
                                                </option>
                                            ))}
                                        </Form.Select>
                                    </div>
                                )}

                                <Button
                                    variant="success"
                                    className="px-4 py-2"
                                    onClick={handleClassificationSettingsUpdate}
                                >
                                    <i className="fas fa-save me-2"></i>
                                    Save Classification Preferences
                                </Button>
                            </div>
                        </Col>

                        {/* Explanation Settings */}
                        <Col xs={12}>
                            <div className="h-100 p-4 rounded bg-white border shadow-sm">
                                <h5 className="mb-4 text-info">
                                    <i className="fas fa-lightbulb me-2"></i>
                                    Explanation Settings
                                </h5>
                                <p className="text-muted mb-4">Select the AI provider for explanation generation</p>

                                <div className="mb-4">
                                    <h6 className="mb-3">Provider Selection</h6>
                                    <ButtonGroup className="d-flex flex-wrap">
                                        <ToggleButton
                                            id="providerex-openai"
                                            type="radio"
                                            variant={providerex === 'openai' ? 'primary' : 'outline-primary'}
                                            name="providerex"
                                            value="openai"
                                            checked={providerex === 'openai'}
                                            onChange={(e) => setProviderex(e.currentTarget.value)}
                                            className="me-2 mb-2"
                                        >
                                            OpenAI
                                        </ToggleButton>
                                        <ToggleButton
                                            id="providerex-groq"
                                            type="radio"
                                            variant={providerex === 'groq' ? 'primary' : 'outline-primary'}
                                            name="providerex"
                                            value="groq"
                                            checked={providerex === 'groq'}
                                            onChange={(e) => setProviderex(e.currentTarget.value)}
                                            className="me-2 mb-2"
                                        >
                                            Groq
                                        </ToggleButton>
                                        <ToggleButton
                                            id="providerex-deepseek"
                                            type="radio"
                                            variant={providerex === 'deepseek' ? 'primary' : 'outline-primary'}
                                            name="providerex"
                                            value="deepseek"
                                            checked={providerex === 'deepseek'}
                                            onChange={(e) => setProviderex(e.currentTarget.value)}
                                            className="me-2 mb-2"
                                        >
                                            Deepseek
                                        </ToggleButton>
                                        <ToggleButton
                                            id="providerex-openrouter"
                                            type="radio"
                                            variant={providerex === 'openrouter' ? 'primary' : 'outline-primary'}
                                            name="providerex"
                                            value="openrouter"
                                            checked={providerex === 'openrouter'}
                                            onChange={(e) => setProviderex(e.currentTarget.value)}
                                            className="me-2 mb-2"
                                        >
                                            Openrouter
                                        </ToggleButton>
                                        <ToggleButton
                                            id="providerex-gemini"
                                            type="radio"
                                            variant={providerex === 'gemini' ? 'primary' : 'outline-primary'}
                                            name="providerex"
                                            value="gemini"
                                            checked={providerex === 'gemini'}
                                            onChange={(e) => setProviderex(e.currentTarget.value)}
                                            className="me-2 mb-2"
                                        >
                                            Gemini
                                        </ToggleButton>
                                        <ToggleButton
                                            id="providerex-ollama"
                                            type="radio"
                                            variant={providerex === 'ollama' ? 'primary' : 'outline-primary'}
                                            name="providerex"
                                            value="ollama"
                                            checked={providerex === 'ollama'}
                                            onChange={(e) => setProviderex(e.currentTarget.value)}
                                            className="mb-2"
                                        >
                                            Ollama
                                        </ToggleButton>
                                    </ButtonGroup>
                                </div>

                                {/* Model Selection */}
                                {(providerex === 'gemini' || providerex === 'groq' || providerex === 'openai' || providerex === 'openrouter' || providerex === 'ollama') && (
                                    <div className="mb-4">
                                        <h6 className="mb-3">Model Selection</h6>
                                        <Form.Select 
                                            value={modelex} 
                                            onChange={(e) => setModelex(e.target.value)}
                                            className="border-0 bg-light"
                                        >
                                            <option value="">-- Select a Model --</option>
                                            {providerex === 'gemini' && geminiModels.map((m) => (
                                                <option key={m.name} value={m.name}>
                                                    {m.name}
                                                </option>
                                            ))}
                                            {providerex === 'groq' && groqModels.map((m) => (
                                                <option key={m.name} value={m.name}>
                                                    {m.name}
                                                </option>
                                            ))}
                                            {providerex === 'openai' && openAIModels.map((m) => (
                                                <option key={m.name} value={m.name}>
                                                    {m.name}
                                                </option>
                                            ))}
                                            {providerex === 'openrouter' && openrouterModels.map((m) => (
                                                <option key={m.name} value={m.name}>
                                                    {m.name}
                                                </option>
                                            ))}
                                            {providerex === 'ollama' && ollamaModels.map((m) => (
                                                <option key={m.name} value={m.name}>
                                                    {m.name}
                                                </option>
                                            ))}
                                        </Form.Select>
                                    </div>
                                )}

                                <Button
                                    variant="info"
                                    className="px-4 py-2"
                                    onClick={handleExplanationSettingsUpdate}
                                >
                                    <i className="fas fa-save me-2"></i>
                                    Save Explanation Preferences
                                </Button>
                            </div>
                        </Col>
                    </Row>
                </Col>
            </Row>
        </Container>
    );
};

export default Settings;