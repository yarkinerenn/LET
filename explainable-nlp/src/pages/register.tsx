import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Form, Button, Alert, Container, Row, Col } from 'react-bootstrap';
import axios from 'axios';

const Register: React.FC = () => {
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const navigate = useNavigate();
    const [openaiApi, setOpenaiApi] = useState(""); // State for OpenAI API Key
    const [grokApi, setGrokApi] = useState("");
    const [deepseekApi, setDeepseekApi] = useState("");
    const [openrouterApi, setOpenrouterApi] = useState("");
    const [geminiApi, setGeminiApi] = useState("");

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        try {
            const response = await axios.post('http://localhost:5000/api/register', {
                username,
                email,
                password,
                openai_api: openaiApi, // Send OpenAI API Key
                grok_api: grokApi,
                deepseek_api: deepseekApi,
                openrouter_api: openrouterApi,
                gemini_api: geminiApi,
            });

            if (response.status === 201) {
                // Registration successful, redirect to login
                navigate('/login');
            }
        } catch (err: any) {
            if (err.response) {
                setError(err.response.data.error || 'Registration failed');
            } else {
                setError('An error occurred during registration');
            }
        }
    };

    return (
        <Container className="py-5">
            <Row className="justify-content-center">
                <Col lg={10} xl={8}>
                    <div className="auth-card">
                        <h2 className="text-center mb-4">Create Account</h2>
                        {error && <Alert variant="danger">{error}</Alert>}

                        <Row className="g-4">
                            {/* Left Column - Basic Registration */}
                            <Col md={6}>
                                <h5 className="mb-3 text-primary">
                                    <i className="fas fa-user me-2"></i>
                                    Account Information
                                </h5>
                                <Form onSubmit={handleSubmit}>
                            <Form.Group className="mb-3">
                                <Form.Label>Username</Form.Label>
                                <Form.Control
                                    type="text"
                                    placeholder="Enter username"
                                    value={username}
                                    onChange={(e) => setUsername(e.target.value)}
                                    required
                                />
                            </Form.Group>

                            <Form.Group className="mb-3">
                                <Form.Label>Email address</Form.Label>
                                <Form.Control
                                    type="email"
                                    placeholder="Enter email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    required
                                />
                            </Form.Group>

                            <Form.Group className="mb-4">
                                <Form.Label>Password</Form.Label>
                                <Form.Control
                                    type="password"
                                    placeholder="Password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    required
                                />
                            </Form.Group>

                            <Button variant="dark" className="w-100 mb-3" type="submit">
                                Create Account
                            </Button>

                            <div className="text-center">
                                <small>
                                    Already have an account?{' '}
                                    <a href="/login" className="text-primary">
                                        Login here
                                    </a>
                                </small>
                            </div>
                                </Form>
                            </Col>

                            {/* Right Column - API Keys */}
                            <Col md={6}>
                                <h5 className="mb-3 text-success">
                                    <i className="fas fa-key me-2"></i>
                                    API Keys (Optional)
                                </h5>
                                <p className="text-muted mb-4">
                                    <i className="fas fa-info-circle me-2"></i>
                                    To use Large Language Models for explanations and classifications, you need at least one API key available. You can add these later in your settings.
                                </p>

                                <Form.Group className="mb-3">
                                    <Form.Label>OpenAI API Key</Form.Label>
                                    <Form.Control
                                        type="text"
                                        placeholder="Enter your OpenAI API key"
                                        value={openaiApi}
                                        onChange={(e) => setOpenaiApi(e.target.value)}
                                    />
                                </Form.Group>

                                {/* Grok API Key Input */}
                                <Form.Group className="mb-3">
                                    <Form.Label>Grok API Key</Form.Label>
                                    <Form.Control
                                        type="text"
                                        placeholder="Enter your Grok API key"
                                        value={grokApi}
                                        onChange={(e) => setGrokApi(e.target.value)}
                                    />
                                </Form.Group>

                                {/* DeepSeek API Key Input */}
                                <Form.Group className="mb-3">
                                    <Form.Label>DeepSeek API Key</Form.Label>
                                    <Form.Control
                                        type="text"
                                        placeholder="Enter your DeepSeek API key"
                                        value={deepseekApi}
                                        onChange={(e) => setDeepseekApi(e.target.value)}
                                    />
                                </Form.Group>

                                {/* OpenRouter API Key Input */}
                                <Form.Group className="mb-3">
                                    <Form.Label>OpenRouter API Key</Form.Label>
                                    <Form.Control
                                        type="text"
                                        placeholder="Enter your OpenRouter API key"
                                        value={openrouterApi}
                                        onChange={(e) => setOpenrouterApi(e.target.value)}
                                    />
                                </Form.Group>

                                {/* Gemini API Key Input */}
                                <Form.Group className="mb-3">
                                    <Form.Label>Gemini API Key</Form.Label>
                                    <Form.Control
                                        type="text"
                                        placeholder="Enter your Gemini API key"
                                        value={geminiApi}
                                        onChange={(e) => setGeminiApi(e.target.value)}
                                    />
                                </Form.Group>
                            </Col>
                        </Row>
                    </div>
                </Col>
            </Row>
        </Container>
    );
};

export default Register;