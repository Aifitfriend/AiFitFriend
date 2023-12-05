// get env data
require('dotenv').config();

// create connection to mongodb
require('../model-db/db');

// get model-db
const User = require('../model-db/model');

const saveDataUser = async (req, res, next) => {
	// get data collection
	const { uid, email, name } = req.user;

	// process input data
	try {
		// check existing user with uid
		const existingUser = await User.findOne({ uid });
		if (!existingUser) {
			const newUser = new User({ uid, email, name });
			await newUser.save();
		}
		next();
	} catch (error) {
		console.error('Error saving user to MongoDB:', error.message);
		res
			.status(500)
			.json({ error: 'Failed to save user data. Error: ' + error.message });
	}
};

module.exports = saveDataUser;
