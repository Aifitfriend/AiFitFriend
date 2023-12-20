package com.capstone.fitfriendapp.view.profile

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import androidx.activity.viewModels
import com.capstone.fitfriendapp.R
import com.capstone.fitfriendapp.databinding.ActivityProfileBinding
import com.capstone.fitfriendapp.view.ViewModelFactory
import com.capstone.fitfriendapp.view.getstarted.GetStartedActivity
import com.capstone.fitfriendapp.view.home.MainActivity

class ProfileActivity : AppCompatActivity() {
    private lateinit var binding: ActivityProfileBinding
    private val viewModel: ProfileViewModel by viewModels {
        ViewModelFactory.getInstance(this)
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityProfileBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val btnBackNavigation: ImageView = findViewById(R.id._navigation_back)
        btnBackNavigation.setOnClickListener {
            val moveIntent = Intent(this@ProfileActivity, MainActivity::class.java)
            startActivity(moveIntent)
        }

        val btnEditActivity: ImageView = findViewById(R.id.edit)
        btnEditActivity.setOnClickListener {
            val moveIntent = Intent(this@ProfileActivity, EditProfileActivity::class.java)
            startActivity(moveIntent)
        }

        val btnLogoutActivity: Button = findViewById(R.id.button_logout)
        btnLogoutActivity.setOnClickListener {
            viewModel.logout()
            signOut()
        }
    }

    private fun signOut() {
        startActivity(Intent(this, GetStartedActivity::class.java))
        finish()
    }
}