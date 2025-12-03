import React, { useState, useEffect } from "react";
import {Container, Row, Col, Form, Button, Alert, ButtonGroup, ToggleButton, Badge} from "react-bootstrap";
import {useProvider} from "../modules/provider";
import axios from "axios";

const Settings = () => {
    const [openaiApi, setOpenaiApi] = useState(""); // Current OpenAI API Key
    const [groqApi, setGroqApi] = useState("");     // Current Groq API Key
    const [deepseekApi, setdeepseekApi] = useState("");
    const [openrouterApi, setopenrouterApi] = useState("");
    const [geminiApi, setGeminiApi] = useState("");
    const [error, setError] = useState("");         // For error messages
    const [success, setSuccess] = useState("");     // For success message
    const [apiKeysStatus, setApiKeysStatus] = useState({
        openai_api: false,
        groq_api: false,
        deepseek_api: false,
        openrouter_api: false,
        gemini_api: false
    });
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

    // Helper function to check if provider has API key
    const hasApiKeyForProvider = (providerName: string): boolean => {
        // Ollama doesn't require an API key (local model)
        if (providerName === 'ollama') {
            return true;
        }
        
        const apiKeyMap: { [key: string]: keyof typeof apiKeysStatus } = {
            'openai': 'openai_api',
            'groq': 'groq_api',
            'deepseek': 'deepseek_api',
            'openrouter': 'openrouter_api',
            'gemini': 'gemini_api'
        };
        
        const apiKeyField = apiKeyMap[providerName];
        return apiKeyField ? apiKeysStatus[apiKeyField] : false;
    };

    // Fetch API keys status and user preferences on component mount
    useEffect(() => {
        const fetchApiKeysStatus = async () => {
            try {
                const response = await fetch("/api/settings/get_api_keys_status", {
                    method: "GET",
                    credentials: 'include',
                });
                if (response.ok) {
                    const data = await response.json();
                    setApiKeysStatus(data);
                    return data; // Return the data for use in fetchPreferences
                }
            } catch (error) {
                console.error("Failed to fetch API keys status:", error);
            }
            return null;
        };
        
        const fetchPreferences = async (keysStatus: typeof apiKeysStatus) => {
            try {
                const response = await fetch("/api/settings/get_preferences", {
                    method: "GET",
                    credentials: 'include',
                });
                if (response.ok) {
                    const data = await response.json();
                    
                    // Helper to check API key with provided status
                    const checkApiKey = (providerName: string): boolean => {
                        if (providerName === 'ollama') return true;
                        const apiKeyMap: { [key: string]: keyof typeof keysStatus } = {
                            'openai': 'openai_api',
                            'groq': 'groq_api',
                            'deepseek': 'deepseek_api',
                            'openrouter': 'openrouter_api',
                            'gemini': 'gemini_api'
                        };
                        const apiKeyField = apiKeyMap[providerName];
                        return apiKeyField ? keysStatus[apiKeyField] : false;
                    };
                    
                    // Set provider and model from preferences only if API key is available
                    if (data.preferred_provider && checkApiKey(data.preferred_provider)) {
                        setProvider(data.preferred_provider);
                        if (data.preferred_model) setModel(data.preferred_model);
                    }
                    if (data.preferred_providerex && checkApiKey(data.preferred_providerex)) {
                        setProviderex(data.preferred_providerex);
                        if (data.preferred_modelex) setModelex(data.preferred_modelex);
                    }
                }
            } catch (error) {
                console.error("Failed to fetch preferences:", error);
            }
        };
        
        // Fetch API keys status first, then preferences with the status
        const initializeSettings = async () => {
            const keysStatus = await fetchApiKeysStatus();
            if (keysStatus) {
                await fetchPreferences(keysStatus);
            }
        };
        
        initializeSettings();
    }, [setProvider, setModel, setProviderex, setModelex]);

    // Reset provider selection if API key is not available
    useEffect(() => {
        if (provider && !hasApiKeyForProvider(provider)) {
            setProvider('');
            setModel('');
        }
    }, [apiKeysStatus, provider, setProvider, setModel]);

    useEffect(() => {
        if (providerex && !hasApiKeyForProvider(providerex)) {
            setProviderex('');
            setModelex('');
        }
    }, [apiKeysStatus, providerex, setProviderex, setModelex]);
    const handleExplanationSettingsUpdate = async () => {
        const payload = {
            preferred_providerex: providerex,
            preferred_modelex: modelex
        };

        try {
            const response = await fetch("/api/settings/update_preferred_explanation", {
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
                // Refresh preferences to show updated selection
                const prefResponse = await fetch("/api/settings/get_preferences", {
                    method: "GET",
                    credentials: 'include',
                });
                if (prefResponse.ok) {
                    const prefData = await prefResponse.json();
                    if (prefData.preferred_providerex) setProviderex(prefData.preferred_providerex);
                    if (prefData.preferred_modelex) setModelex(prefData.preferred_modelex);
                }
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
            const response = await fetch("/api/settings/update_preferred_classification", {
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
                // Refresh preferences to show updated selection
                const prefResponse = await fetch("/api/settings/get_preferences", {
                    method: "GET",
                    credentials: 'include',
                });
                if (prefResponse.ok) {
                    const prefData = await prefResponse.json();
                    if (prefData.preferred_provider) setProvider(prefData.preferred_provider);
                    if (prefData.preferred_model) setModel(prefData.preferred_model);
                }
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
        if (!openaiApi && !groqApi && !deepseekApi && !openrouterApi && !geminiApi) {
            setError("Please enter at least one API key to update. Leave fields empty if you don't want to change existing keys.");
            console.log("Error: All API fields are empty.");
            return;
        }

        console.log("API keys provided:", { openaiApi, groqApi });

        setError(""); // Clear previous errors
        setSuccess(""); // Clear previous success messages

        // Prepare the request payload (only include non-empty values)
        const payload: { openai_api?: string; groq_api?: string; deepseek_api?: string; openrouter_api?: string; gemini_api?: string } = {};
        if (openaiApi) payload.openai_api = openaiApi;
        if (groqApi) payload.groq_api = groqApi;
        if (deepseekApi) payload.deepseek_api = deepseekApi;
        if (openrouterApi) payload.openrouter_api = openrouterApi;
        if (geminiApi) payload.gemini_api = geminiApi;


        console.log("Sending request with payload:", payload);

        try {
            const response = await fetch("/api/settings/update_api_keys", {
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
                setGroqApi("");
                setdeepseekApi("");
                setopenrouterApi("");
                setGeminiApi("");
                // Refresh API keys status
                const statusResponse = await fetch("/api/settings/get_api_keys_status", {
                    method: "GET",
                    credentials: 'include',
                });
                if (statusResponse.ok) {
                    const statusData = await statusResponse.json();
                    setApiKeysStatus(statusData);
                }
            } else {
                console.error("Error updating API keys:", result);
                setError(result.error || "An error occurred.");
            }
        } catch (error) {
            console.error("Fetch error:", error);
            setError("Failed to connect to the server.");
        }
    };

    const handleDeleteApiKey = async (keyType: string) => {
        const keyTypeNames: { [key: string]: string } = {
            'openai_api': 'OpenAI',
            'groq_api': 'Groq',
            'deepseek_api': 'DeepSeek',
            'openrouter_api': 'OpenRouter',
            'gemini_api': 'Gemini'
        };
        
        const keyName = keyTypeNames[keyType] || keyType;
        
        if (!window.confirm(`Are you sure you want to delete your ${keyName} API key?`)) {
            return;
        }

        try {
            const response = await axios.post(
                "/api/settings/delete_api_key",
                { key_type: keyType },
                { withCredentials: true }
            );

            if (response.status === 200) {
                setSuccess(`${keyName} API key deleted successfully`);
                setError("");
                
                // Refresh API keys status
                const statusResponse = await fetch("/api/settings/get_api_keys_status", {
                    method: "GET",
                    credentials: 'include',
                });
                if (statusResponse.ok) {
                    const statusData = await statusResponse.json();
                    setApiKeysStatus(statusData);
                }
                
                // Reset provider selection if the deleted key was being used
                const providerMap: { [key: string]: string } = {
                    'openai_api': 'openai',
                    'groq_api': 'groq',
                    'deepseek_api': 'deepseek',
                    'openrouter_api': 'openrouter',
                    'gemini_api': 'gemini'
                };
                const affectedProvider = providerMap[keyType];
                if (affectedProvider && provider === affectedProvider) {
                    setProvider('');
                    setModel('');
                }
                if (affectedProvider && providerex === affectedProvider) {
                    setProviderex('');
                    setModelex('');
                }
            } else {
                setError(response.data?.error || "Failed to delete API key");
            }
        } catch (error: any) {
            console.error("Delete API key error:", error);
            setError(error.response?.data?.error || "Failed to delete API key");
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
                        <h5 className="mb-2 text-primary">
                            <i className="fas fa-key me-2"></i>
                            API Keys
                        </h5>
                        <p className="text-muted mb-3">Configure your API keys for different providers</p>
                        <Alert variant="info" className="py-2 mb-3">
                            <i className="fas fa-shield-alt me-2"></i>
                            <strong>Security:</strong> All API keys are encrypted and securely stored.
                        </Alert>
                        <Alert variant="info" className="py-2">
                            To compute trustworthiness metrics, please enter a Groq API key.
                        </Alert>

                        <Form onSubmit={handleSubmit}>
                            <Form.Group className="mb-3">
                                <Form.Label className="fw-semibold d-flex align-items-center justify-content-between">
                                    <span>OpenAI API Key</span>
                                    <div className="d-flex align-items-center gap-2">
                                        {apiKeysStatus.openai_api && (
                                            <>
                                                <Badge bg="success" className="ms-2">
                                                    <i className="fas fa-check-circle me-1"></i>
                                                    Set
                                                </Badge>
                                                <Button
                                                    variant="outline-danger"
                                                    size="sm"
                                                    onClick={() => handleDeleteApiKey('openai_api')}
                                                    title="Delete API key"
                                                    style={{ padding: '2px 8px', fontSize: '0.75rem' }}
                                                >
                                                    <i className="fas fa-trash-alt"></i>
                                                </Button>
                                            </>
                                        )}
                                    </div>
                                </Form.Label>
                                <Form.Control
                                    type="password"
                                    placeholder={apiKeysStatus.openai_api ? "API key already set (enter new key to update)" : "Enter your OpenAI API key"}
                                    value={openaiApi}
                                    onChange={(e) => setOpenaiApi(e.target.value)}
                                    className="border-0 bg-light"
                                />
                            </Form.Group>

                            <Form.Group className="mb-3">
                                <Form.Label className="fw-semibold d-flex align-items-center justify-content-between">
                                    <span>Groq API Key</span>
                                    <div className="d-flex align-items-center gap-2">
                                        {apiKeysStatus.groq_api && (
                                            <>
                                                <Badge bg="success" className="ms-2">
                                                    <i className="fas fa-check-circle me-1"></i>
                                                    Set
                                                </Badge>
                                                <Button
                                                    variant="outline-danger"
                                                    size="sm"
                                                    onClick={() => handleDeleteApiKey('groq_api')}
                                                    title="Delete API key"
                                                    style={{ padding: '2px 8px', fontSize: '0.75rem' }}
                                                >
                                                    <i className="fas fa-trash-alt"></i>
                                                </Button>
                                            </>
                                        )}
                                    </div>
                                </Form.Label>
                                <Form.Control
                                    type="password"
                                    placeholder={apiKeysStatus.groq_api ? "API key already set (enter new key to update)" : "Enter your Groq API key"}
                                    value={groqApi}
                                    onChange={(e) => setGroqApi(e.target.value)}
                                    className="border-0 bg-light"
                                />
                            </Form.Group>

                            <Form.Group className="mb-3">
                                <Form.Label className="fw-semibold d-flex align-items-center justify-content-between">
                                    <span>DeepSeek API Key</span>
                                    <div className="d-flex align-items-center gap-2">
                                        {apiKeysStatus.deepseek_api && (
                                            <>
                                                <Badge bg="success" className="ms-2">
                                                    <i className="fas fa-check-circle me-1"></i>
                                                    Set
                                                </Badge>
                                                <Button
                                                    variant="outline-danger"
                                                    size="sm"
                                                    onClick={() => handleDeleteApiKey('deepseek_api')}
                                                    title="Delete API key"
                                                    style={{ padding: '2px 8px', fontSize: '0.75rem' }}
                                                >
                                                    <i className="fas fa-trash-alt"></i>
                                                </Button>
                                            </>
                                        )}
                                    </div>
                                </Form.Label>
                                <Form.Control
                                    type="password"
                                    placeholder={apiKeysStatus.deepseek_api ? "API key already set (enter new key to update)" : "Enter your Deepseek API key"}
                                    value={deepseekApi}
                                    onChange={(e) => setdeepseekApi(e.target.value)}
                                    className="border-0 bg-light"
                                />
                            </Form.Group>

                            <Form.Group className="mb-3">
                                <Form.Label className="fw-semibold d-flex align-items-center justify-content-between">
                                    <span>Openrouter API Key</span>
                                    <div className="d-flex align-items-center gap-2">
                                        {apiKeysStatus.openrouter_api && (
                                            <>
                                                <Badge bg="success" className="ms-2">
                                                    <i className="fas fa-check-circle me-1"></i>
                                                    Set
                                                </Badge>
                                                <Button
                                                    variant="outline-danger"
                                                    size="sm"
                                                    onClick={() => handleDeleteApiKey('openrouter_api')}
                                                    title="Delete API key"
                                                    style={{ padding: '2px 8px', fontSize: '0.75rem' }}
                                                >
                                                    <i className="fas fa-trash-alt"></i>
                                                </Button>
                                            </>
                                        )}
                                    </div>
                                </Form.Label>
                                <Form.Control
                                    type="password"
                                    placeholder={apiKeysStatus.openrouter_api ? "API key already set (enter new key to update)" : "Enter your Openrouter API key"}
                                    value={openrouterApi}
                                    onChange={(e) => setopenrouterApi(e.target.value)}
                                    className="border-0 bg-light"
                                />
                            </Form.Group>

                            <Form.Group className="mb-4">
                                <Form.Label className="fw-semibold d-flex align-items-center justify-content-between">
                                    <span>Gemini API Key</span>
                                    <div className="d-flex align-items-center gap-2">
                                        {apiKeysStatus.gemini_api && (
                                            <>
                                                <Badge bg="success" className="ms-2">
                                                    <i className="fas fa-check-circle me-1"></i>
                                                    Set
                                                </Badge>
                                                <Button
                                                    variant="outline-danger"
                                                    size="sm"
                                                    onClick={() => handleDeleteApiKey('gemini_api')}
                                                    title="Delete API key"
                                                    style={{ padding: '2px 8px', fontSize: '0.75rem' }}
                                                >
                                                    <i className="fas fa-trash-alt"></i>
                                                </Button>
                                            </>
                                        )}
                                    </div>
                                </Form.Label>
                                <Form.Control
                                    type="password"
                                    placeholder={apiKeysStatus.gemini_api ? "API key already set (enter new key to update)" : "Enter your Gemini API key"}
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
                                            disabled={!hasApiKeyForProvider('openai')}
                                            title={!hasApiKeyForProvider('openai') ? 'API key required' : ''}
                                        >
                                            OpenAI {!hasApiKeyForProvider('openai') && <Badge bg="secondary" className="ms-1" style={{ fontSize: '0.65rem' }}>API Key Required</Badge>}
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
                                            disabled={!hasApiKeyForProvider('groq')}
                                            title={!hasApiKeyForProvider('groq') ? 'API key required' : ''}
                                        >
                                            Groq {!hasApiKeyForProvider('groq') && <Badge bg="secondary" className="ms-1" style={{ fontSize: '0.65rem' }}>API Key Required</Badge>}
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
                                            disabled={!hasApiKeyForProvider('deepseek')}
                                            title={!hasApiKeyForProvider('deepseek') ? 'API key required' : ''}
                                        >
                                            Deepseek {!hasApiKeyForProvider('deepseek') && <Badge bg="secondary" className="ms-1" style={{ fontSize: '0.65rem' }}>API Key Required</Badge>}
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
                                            disabled={!hasApiKeyForProvider('openrouter')}
                                            title={!hasApiKeyForProvider('openrouter') ? 'API key required' : ''}
                                        >
                                            Openrouter {!hasApiKeyForProvider('openrouter') && <Badge bg="secondary" className="ms-1" style={{ fontSize: '0.65rem' }}>API Key Required</Badge>}
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
                                            disabled={!hasApiKeyForProvider('gemini')}
                                            title={!hasApiKeyForProvider('gemini') ? 'API key required' : ''}
                                        >
                                            Gemini {!hasApiKeyForProvider('gemini') && <Badge bg="secondary" className="ms-1" style={{ fontSize: '0.65rem' }}>API Key Required</Badge>}
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
                                            disabled={!hasApiKeyForProvider('openai')}
                                            title={!hasApiKeyForProvider('openai') ? 'API key required' : ''}
                                        >
                                            OpenAI {!hasApiKeyForProvider('openai') && <Badge bg="secondary" className="ms-1" style={{ fontSize: '0.65rem' }}>API Key Required</Badge>}
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
                                            disabled={!hasApiKeyForProvider('groq')}
                                            title={!hasApiKeyForProvider('groq') ? 'API key required' : ''}
                                        >
                                            Groq {!hasApiKeyForProvider('groq') && <Badge bg="secondary" className="ms-1" style={{ fontSize: '0.65rem' }}>API Key Required</Badge>}
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
                                            disabled={!hasApiKeyForProvider('deepseek')}
                                            title={!hasApiKeyForProvider('deepseek') ? 'API key required' : ''}
                                        >
                                            Deepseek {!hasApiKeyForProvider('deepseek') && <Badge bg="secondary" className="ms-1" style={{ fontSize: '0.65rem' }}>API Key Required</Badge>}
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
                                            disabled={!hasApiKeyForProvider('openrouter')}
                                            title={!hasApiKeyForProvider('openrouter') ? 'API key required' : ''}
                                        >
                                            Openrouter {!hasApiKeyForProvider('openrouter') && <Badge bg="secondary" className="ms-1" style={{ fontSize: '0.65rem' }}>API Key Required</Badge>}
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
                                            disabled={!hasApiKeyForProvider('gemini')}
                                            title={!hasApiKeyForProvider('gemini') ? 'API key required' : ''}
                                        >
                                            Gemini {!hasApiKeyForProvider('gemini') && <Badge bg="secondary" className="ms-1" style={{ fontSize: '0.65rem' }}>API Key Required</Badge>}
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