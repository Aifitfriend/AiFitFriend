const admin = require('firebase-admin');
require('dotenv').config();

//initializing firebase setup
admin.initializeApp({
	credential: admin.credential.cert({
		type: process.env.TYPE,
		project_id: process.env.PROJECT_ID,
		private_key_id: process.env.PRIVATE_KEY_ID,
		private_key: process.env.PRIVATE_KEY.replace(/\\n/g, '\n'),
		client_email: process.env.CLIENT_EMAIL,
		client_id: process.env.CLIENT_ID,
		auth_uri: process.env.AUTH_URI,
		token_uri: process.env.TOKEN_URI,
		auth_provider_x509_cert_url: process.env.AUTH_PROVIDER_X509_CERT_URL,
		client_x509_cert_url: process.env.CLIENT_X509_CERT_URL,
		universe_domain: process.env.UNIVERSE_DOMAIN,
	}),
});

// get token from header request
const getTokenFromHeaderUser = (req) => {
	const authorization = req.get('Authorization');
	if (authorization && authorization.split(' ')[0] === 'Bearer') {
		return authorization.split(' ')[1];
	}
	return null;
};

// Authentication user
const userAuth = async (token, res, next) => {
	try {
		const decodedToken = await admin.auth().verifyIdToken(token);
		const uid = decodedToken.uid;
		console.log('Successfully verified token for user:', uid);
		next();
	} catch (error) {
		console.error('Error verifying token:', error);
		res.status(403).json({ status: 'error', message: 'Forbidden' });
	}
};

const getResponseAuth = async (req, res, next) => {
	const token = getTokenFromHeaderUser(req);
	// Check token user
	if (!token) {
		return res
			.status(403)
			.json({ status: 'error', message: 'Forbidden: Token not provided' });
	} else {
		try {
			await userAuth(token, res, next);
		} catch (error) {
			console.error('Error in userAuth middleware:', error);
			return res
				.status(500)
				.json({ status: 'error', message: 'Internal Server Error' });
		}
	}
};

module.exports = { getResponseAuth };
