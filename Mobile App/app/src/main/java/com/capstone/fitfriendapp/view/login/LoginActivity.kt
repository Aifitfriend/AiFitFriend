package com.capstone.fitfriendapp.view.login

import android.content.Intent
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AlertDialog
import com.capstone.fitfriendapp.R
import com.capstone.fitfriendapp.data.pref.ResultState
import com.capstone.fitfriendapp.data.pref.UserModel
import com.capstone.fitfriendapp.databinding.ActivityLoginBinding
import com.capstone.fitfriendapp.view.ViewModelFactory
import com.capstone.fitfriendapp.view.home.MainActivity
import kotlinx.coroutines.flow.map

class LoginActivity : AppCompatActivity() {
    private lateinit var binding: ActivityLoginBinding
    private val viewModel : LoginViewModel by viewModels {
        ViewModelFactory.getInstance(this)
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityLoginBinding.inflate(layoutInflater)
        setContentView(binding.root)

        viewModel.getSessionIsLogin().map{
            if (it.isLogin){
               startMainActivity()
            }
        }

        setupView()
        setupAction()
    }

    private fun setupView() {
        @Suppress("DEPRECATION")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.insetsController?.hide(WindowInsets.Type.statusBars())
        } else {
            window.setFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN
            )
        }
        supportActionBar?.hide()
    }
    private fun setupAction() {
        binding.loginButton.setOnClickListener {
            val email = binding.edLoginEmail.text.toString()
            val password = binding.edLoginPassword.text.toString()

            if (email.isEmpty()) {
                binding.loginEmail.error = getString(R.string.email_rule)
            } else if (password.isEmpty()) {
                binding.loginPassword.error = getString(R.string.password_rule)
            }

            viewModel.login(email, password).observe(this) { response ->
                when(response) {
                    is ResultState.Loading -> {
                        showLoading(true)
                    }
                    is ResultState.Success -> {
                        response.data.message.let {
                            viewModel.saveSession(
                                UserModel(
                                    email, response.data.loginResult.token
                                )
                            )
                            showLoading(false)
                            showLoginSuccessDialog()
                        }
                    }
                    is ResultState.Error -> {
                        showLoading(false)
                        showLoginErrorToast()
                    }
                }
            }
        }
    }

    private fun showLoginSuccessDialog(){
        AlertDialog.Builder(this).apply {
            setTitle("Yeah!")
            setMessage("Anda berhasil login!")
            setPositiveButton("Lanjut") { _, _ ->
                startMainActivity()
            }
            create()
            show()
        }
    }

    private fun showLoginErrorToast() {
        Toast.makeText(
            this, "Login gagal. Silahkan coba lagi.",
            Toast.LENGTH_SHORT).show()
    }

    private fun startMainActivity() {
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_CLEAR_TASK or Intent.FLAG_ACTIVITY_NEW_TASK
        }
        startActivity(intent)
        finish()
    }

    private fun showLoading(state: Boolean) {
        binding.progressBar.visibility = if (state) View.VISIBLE else View.GONE
    }

}