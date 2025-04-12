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

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        try {
            const response = await axios.post('http://localhost:5000/api/register', {
                username,
                email,
                password,
                openai_api: openaiApi, // Send OpenAI API Key
                grok_api: grokApi,
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
                <Col md={6} lg={4}>
                    <div className="auth-card">
                        <h2 className="text-center mb-4">Create Account</h2>
                        {error && <Alert variant="danger">{error}</Alert>}

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

                            <Button variant="primary" className="w-100 mb-3" type="submit">
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
                    </div>
                </Col>
            </Row>
        </Container>
    );
};

export default Register;